# @title
import ast
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import argparse
import re
import warnings
import logging
import torch
import random
import numpy as np
from analyze import *
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
     RobertaTokenizer, RobertaForMaskedLM, pipeline,
     T5Config, T5ForConditionalGeneration, AutoTokenizer
)
import keyword
import threading
import queue
import time

warnings.filterwarnings("ignore")
logging.getLogger('transformers').setLevel(logging.ERROR)




parser = argparse.ArgumentParser(description='Extracting memorized content from code snippets')
parser.add_argument('--prog_lang', type=str, default='python', help='Programming language of the code snippets',choices=['python','java','javascript','ruby'])
parser.add_argument('--mask_ratio', type=float, default=0.75, help='Masking ratio for the code snippets')
parser.add_argument('--output_dir', type=str, default='output', help='Output directory for the extracted content')
parser.add_argument('--device', type=str, default='cuda', help='Device to run the extraction process on', choices=['cpu','cuda'])
parser.add_argument('--autoregressive_model_name', type=str, default='Salesforce/codet5-large', help='Name of the autoregressive model to use for extracting content')
parser.add_argument('--non_autoregressive_model_name', type=str, default='microsoft/codebert-base-mlm', help='Name of the non-autoregressive model to use for extracting content')
parser.add_argument('--quantization', default=False, help='Whether to quantize the model or not')
args = parser.parse_args()

print(f'Using device: {args.device}')
print(f'Programming language: {args.prog_lang}')
print(f'Masking ratio: {args.mask_ratio}')
print(f'Output directory: {args.output_dir}')
print(f'Autoregressive model: {args.autoregressive_model_name}')
print(f'Non-autoregressive model: {args.non_autoregressive_model_name}')
print(f'Quantization: {args.quantization}')
args.quantization = bool(args.quantization)
print(f'Quantization: {type(args.quantization)}')

autoregressive_model_filename = args.output_dir + f'/autoregressive_model_{args.prog_lang}_{args.quantization}.txt'
non_autoregressive_model_filename = args.output_dir + f'/non_autoregressive_model_{args.prog_lang}_{args.quantization}.txt'

analyze_code_function = {'python': analyze_python_code, 'java': analyze_java_code, 'javascript': analyze_javascript_code, 'ruby': analyze_ruby_code}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print('Loading model, dataset and tokenizer')
if args.output_dir in os.listdir():
    print('Output directory already exists')
else:
    os.mkdir(args.output_dir)
model = RobertaForMaskedLM.from_pretrained(args.non_autoregressive_model_name).to(device)
tokenizer = RobertaTokenizer.from_pretrained(args.non_autoregressive_model_name)
#fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Load dataset
prog_lang = args.prog_lang

#dataset = load_dataset("code_search_net", prog_lang)
from unittest.mock import patch

# Function to simulate the input prompt
def mock_input(prompt):
    print(prompt)
    return "y"

# Use the mock to replace input function in the dataset loading context
with patch('builtins.input', mock_input):
    
    dataset = load_dataset("code_search_net", prog_lang)

# Load T5 model
T5_PATH = args.autoregressive_model_name
t5_tokenizer = AutoTokenizer.from_pretrained(T5_PATH)
t5_config = T5Config.from_pretrained(T5_PATH)
t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(device)

if args.quantization:
    t5_mlm = torch.quantization.quantize_dynamic(t5_mlm, {torch.nn.Linear}, dtype=torch.qint8)
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    print('Model quantized successfully')

print('Model, dataset and tokenizer loaded successfully')



builtin_functions = [name for name, obj in __builtins__.__dict__.items() if callable(obj)]


python_keywords = keyword.kwlist


def mask_code(code, consider = ['variable_names','variable_values','strings','comments'],mask_ratio=1,count=0,mask_count=1000):
    if count>5:
        raise Exception("Error in masking code")

    ocode_lines = code.split('\n')
    code_lines = ocode_lines[len(ocode_lines)//4:]
    masked_lines = []
    masked_values=[]
    t5_masked_lines=[]
    result_dict = analyze_code_function[args.prog_lang](code)
    variable_set=[]
    if 'variable_names' in consider:
        variable_set.extend(result_dict['variable_names'])
    if 'variable_values' in consider:
        variable_set.extend(result_dict['variable_values'])
    if 'strings' in consider:
        variable_set.extend(' '.join(result_dict['strings']).split())
    if 'comments' in consider:
        variable_set.extend(' '.join(result_dict['comments']).split())

    counter = 0
    flag =True
    random.seed(42)
    for line in code_lines:
        tokens = re.split(r'(\W+)', line)
        t5_tokens = tokens.copy()
        if flag:
            for i, token in enumerate(tokens):
                t = False
                if random.random() < mask_ratio:
                    if len(masked_values)>=mask_count:
                        flag=False
                        break


                    if token in result_dict['variable_names']:
                        t = True
                        masked_values.append((token, 'variable_names'))
                    elif token in result_dict['variable_values']:
                        t = True
                        masked_values.append((token, 'variable_values'))
                    elif token in ' '.join(result_dict['strings']).split():
                        t = True
                        masked_values.append((token, 'strings'))
                    elif token in ' '.join(result_dict['comments']).split():
                        t = True
                        masked_values.append((token, 'comments'))

                    if t:
                        tokens[i] = '<mask>'
                        t5_tokens[i] = f'<extra_id_{counter}>'
                        counter += 1

        masked_line = ''.join(tokens)
        t5_masked_line = ''.join(t5_tokens)
        masked_lines.append(masked_line)
        t5_masked_lines.append(t5_masked_line)


    if len(masked_values)==0:
        return mask_code(code,consider=consider,mask_ratio=mask_ratio,count=count+1)

    masked_lines = ocode_lines[:len(ocode_lines)//4] + masked_lines
    t5_masked_lines = ocode_lines[:len(ocode_lines)//4] + t5_masked_lines
    return '\n'.join(masked_lines) ,masked_values, '\n'.join(t5_masked_lines)

import time

def t5_get_code(code_snippet):
    max_length = t5_tokenizer.model_max_length
    predicted_values = []


    code_lines = code_snippet.split("\n")
    current_chunk = code_lines[0] + "\n"
    for line in code_lines:
        if len(t5_tokenizer.tokenize(current_chunk + line)) < max_length-5:
            current_chunk += line + "\n"
        else:
            if current_chunk:
                predicted_values.extend(t5_predict_chunk(current_chunk.strip(), t5_tokenizer, t5_mlm))
            current_chunk = code_lines[0]+ line + "\n"

    if current_chunk.strip():
        predicted_values.extend(t5_predict_chunk(current_chunk.strip(), t5_tokenizer, t5_mlm))

    return predicted_values


def t5_predict_chunk(code_snippet, t5_tokenizer, t5_mlm):
    mask_indeices = [f"<extra_id_{i}>" for i in range(code_snippet.count("<mask>"))]
    for i in mask_indeices:
        code_snippet = code_snippet.replace("<mask>",i,1)

    input_ids = t5_tokenizer.encode(code_snippet, return_tensors='pt').to(device)

    if len(input_ids[0]) > t5_tokenizer.model_max_length:
        raise Exception("Input length exceeds the model's maximum length codeT5")

    # Generate the output (estimated masked values)
    start_time = time.time()
    output_ids = t5_mlm.generate(input_ids)
    #print('Time taken T5: ',time.time()-start_time)

    # Decode the output to get the estimated code
    estimated_code = t5_tokenizer.decode(output_ids[0])
    #print(estimated_code)
    # Find and extract the masked values

    predicted_values = [i[2:] for i in estimated_code.split('<extra_id_')[1:]]
    for i in range(len(predicted_values)):
        code_snippet = code_snippet.replace(f'<extra_id_{i}>',predicted_values[i])

    if '<extra_id_' in code_snippet:
        for i in range(len(predicted_values),len(mask_indeices)):
            code_snippet = code_snippet.replace(mask_indeices[i],'<mask>')

        temp = t5_get_code(code_snippet)
        predicted_values.extend(temp)

    return predicted_values


def get_code(masked_code):
    # Replace <mask> with the tokenizer's mask token
    masked_code = masked_code.replace("<mask>", tokenizer.mask_token)
    max_length = tokenizer.model_max_length
    #model.config.max_position_embeddings - 2  # Adjust for special tokens
    predicted_tokens = []
    total_masks = masked_code.count(tokenizer.mask_token)

    # Split the input code into smaller chunks ensuring masks are not split
    code_lines = masked_code.split("\n")
    current_chunk = code_lines[0] + "\n"

    for line in code_lines:
        # Check if adding the current line exceeds the max length
        if len(tokenizer.tokenize(current_chunk + line)) < max_length:
            current_chunk += line + "\n"
        else:
            if current_chunk:
                predicted_tokens.extend(predict_chunk(current_chunk.strip(), tokenizer, model))
            current_chunk = code_lines[0]+line + "\n"

    # Add the last chunk
    if current_chunk.strip():
        predicted_tokens.extend(predict_chunk(current_chunk.strip(), tokenizer, model))

    # Check if all masks are filled
    all_masks_filled = len(predicted_tokens) == total_masks

    return predicted_tokens #, all_masks_filled

def predict_chunk(chunk, tokenizer, model):
    inputs = tokenizer(chunk, return_tensors="pt").to(device)

    if inputs.input_ids.shape[1] > tokenizer.model_max_length:
        raise Exception("Input length exceeds the model's maximum length codeBERT")
    #print(len(inputs.input_ids))
    mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
    predicted_tokens = []

    with torch.no_grad():
        start_time = time.time()
        outputs = model(**inputs)
        #print('Time taken CodeBERT: ',time.time()-start_time)
        predictions = outputs.logits

    for mask_index in mask_token_index:
        mask_predictions = predictions[0, mask_index].topk(1)
        predicted_token_id = mask_predictions.indices.item()
        predicted_token = tokenizer.decode([predicted_token_id]).strip()
        predicted_tokens.append(predicted_token)

    return predicted_tokens

def syntax_mask_code(code, mask_ratio=0.5):

    ocode_lines = code.split('\n')
    code_lines = ocode_lines[len(ocode_lines)//4:]
    masked_lines = []
    masked_values=[]
    t5_masked_lines=[]
    counter = 0
    for line in code_lines:
        tokens = re.split('(\W+)', line)
        t5_tokens = re.split('(\W+)', line)
        for i in range(len(tokens)):
            temp = random.random() < mask_ratio
            if temp and (tokens[i] in builtin_functions + python_keywords ):
                masked_values.append(tokens[i])
                tokens[i] = '<mask>'
                t5_tokens[i] = f'<extra_id_{counter}>'
                counter+=1

        masked_line = ''.join(tokens)
        t5_masked_line = ''.join(t5_tokens)
        masked_lines.append(masked_line)
        t5_masked_lines.append(t5_masked_line)


    masked_lines = ocode_lines[:len(ocode_lines)//4] + masked_lines
    t5_masked_lines = ocode_lines[:len(ocode_lines)//4] + t5_masked_lines
    if len(masked_values)==0:
        mask_code(code)
    return '\n'.join(masked_lines) ,masked_values, '\n'.join(t5_masked_lines)


def compare_metrics(references, predictions):

    score ={'variable_names':0,'variable_values':0,'strings':0,'comments':0}
    ref = {'variable_names':0,'variable_values':0,'strings':0,'comments':0}
    li={'variable_names':[],'variable_values':[],'strings':[],'comments':[]}
    for i in range(len(references)):
        if references[i][0]==predictions[i]:
            score[references[i][1]]+=1
            li[references[i][1]].append(predictions[i])
            #li.append(references[i])
        ref[references[i][1]]+=1
    return {
        'total':sum(score.values())/max(sum(ref.values()),1),
        'variable_names':score['variable_names']/max(ref['variable_names'],1),
        'variable_values':score['variable_values']/max(ref['variable_values'],1),
        'strings':score['strings']/max(ref['strings'], 1),
        'comments':score['comments']/max(ref['comments'], 1)
    },li,sum(score.values())


class AsyncWriter:
    def __init__(self, filename):
        self.filename = filename
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._writer_thread)
        self.thread.start()

    def write(self, data):
        self.queue.put(data)

    def close(self):
        self.stop_event.set()
        self.thread.join()

    def _writer_thread(self):
        with open(self.filename, 'w') as f:
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    data = self.queue.get(timeout=0.1)
                    f.write(data + '\n')
                    f.flush()
                except queue.Empty:
                    continue


print('initiating extraction process...')
gt,pr1 = [],[]
total_match,total_pred=0,0
t5_total_match,t5_total_pred=0,0
bert_res,t5_res = {'variable_names':[],'variable_values':[],'strings':[],'comments':[]},{'variable_names':[],'variable_values':[],'strings':[],'comments':[]}
writer1 = AsyncWriter(non_autoregressive_model_filename)
writer2 = AsyncWriter(autoregressive_model_filename)
for i in range(4000):
    print(f"Extracting memorized content for samples {100*i} to {(i+1)*100}...")
    test = random.randint(0,4000)
    writer1.write(f"Extracting memorized content for samples {100*i} to {(i+1)*100}...")
    writer2.write(f"Extracting memorized content for samples {100*i} to {(i+1)*100}...")
    for instance in tqdm(dataset['train'].select(range(100*i,(i+1)*100))):

        try:
            code = instance['func_code_string']

            masked_code,masked_values,t5_masked_code = mask_code(code,mask_ratio=args.mask_ratio)
            # print(len(masked_code))
            # print(masked_code)

            top_tokens = get_code(masked_code)
            t5_top_tokens = t5_get_code(masked_code)
            res = compare_metrics(masked_values,top_tokens)
            total_match+=res[2]
            total_pred+=len(top_tokens)
            gt.append(res[0]['total'])
            # print(len(top_tokens),len(masked_values),len(t5_top_tokens))
            # print(top_tokens)
            # print(t5_top_tokens)
            res1 = compare_metrics(masked_values,t5_top_tokens)
            t5_total_match+=res1[2]
            t5_total_pred+=len(t5_top_tokens)
            pr1.append(res1[0]['total'])

            writer1.write('-'*50 + '\n')
            #writer1.write(f'metrics: {res[0]}')
            writer1.write(f"Exact match: {res[1]}")
            writer2.write('-'*50 + '\n')
            #writer2.write(f'metrics: {res1[0]}')
            writer2.write(f"Exact match: {res1[1]}")

            for key in bert_res:
                bert_res[key].append(res[0][key])
                t5_res[key].append(res1[0][key])
        except Exception as e:
            continue


    print('codeBERT: ')
    print('exact match: ',np.array(gt).mean())
    print('variable_names: ',np.array(bert_res['variable_names']).mean())
    print('variable_values: ',np.array(bert_res['variable_values']).mean())
    print('strings: ',np.array(bert_res['strings']).mean())
    print('comments: ',np.array(bert_res['comments']).mean())
    print('total match: ',total_match)
    print('total pred: ',total_pred)
    print('T5: ')
    print('exact match: ',np.array(pr1).mean())
    print('variable_names: ',np.array(t5_res['variable_names']).mean())
    print('variable_values: ',np.array(t5_res['variable_values']).mean())
    print('strings: ',np.array(t5_res['strings']).mean())
    print('comments: ',np.array(t5_res['comments']).mean())
    print('exact match: ',t5_total_match)
    print('total pred: ',t5_total_pred)
    print('-'*50)

writer1.close()
writer2.close()

print('Extraction process completed successfully')
