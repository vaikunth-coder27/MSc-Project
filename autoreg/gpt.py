import ast
import re
import warnings
import logging
import torch
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import os
import transformers
import argparse
import random
from codebleu import calc_codebleu


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

warnings.filterwarnings("ignore")
logging.getLogger('transformers').setLevel(logging.ERROR)

parser = argparse.ArgumentParser(description='Extracting memorized content from code snippets')
parser.add_argument('--prog_lang', type=str, default='python', help='Programming language of the code snippets',choices=['python','java','javascript','ruby'])
parser.add_argument('--device', type=str, default='cpu', help='Device to run the extraction process on', choices=['cpu','cuda'])
parser.add_argument('--model_name', type=str, default='AISE-TUDelft/CodeGPT-Multilingual', help='Name of the autoregressive model to use for extracting content')
parser.add_argument('--quantization', type=bool, default=False, help='Whether to quantize the model or not')
args = parser.parse_args()

print(f'Using device: {args.device}')
print(f'Programming language: {args.prog_lang}')
print(f'Model name: {args.model_name}')
print(f'Quantization: {args.quantization}')


device = args.device 
model_name = args.model_name
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(device)

if args.quantization:
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    print('Model quantized successfully')

prog_language = args.prog_lang
dataset = load_dataset("code_search_net", prog_language)

print('Model, dataset and tokenizer loaded successfully')

print('initiating extraction process...')

bleu={'codebleu':np.array([]),'ngram_match_score':np.array([]),'weighted_ngram_match_score':np.array([]),'syntax_match_score':np.array([]),'dataflow_match_score':np.array([])}

max_tokens = tokenizer.model_max_length
tokenizer.pad_token = tokenizer.eos_token


for i in range(4000):
    print(f"Extracting memorized content for samples {100*i} to {(i+1)*100}...")
    test = random.randint(0, 4000)
    for instance in tqdm(dataset['train'].select(range(100*i, (i+1)*100))):
        code = instance['func_code_string']
        prefix = code[:len(code)//2]
        suffix = code[len(code)//2:]
        try:
            input_ids = tokenizer.encode(prefix, return_tensors="pt").to(device)
            max_length = min(model.config.n_positions, input_ids.shape[1] + 500)
            output_ids = model.generate(input_ids, max_length=max_length, temperature=0.7, top_k=50, top_p=0.95, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)        
            bleu1 = calc_codebleu([suffix],[generated_text],lang=prog_language, weights=(0.25, 0.25, 0.25, 0.25))

            for key in bleu.keys():
                bleu[key] = np.append(bleu[key],bleu1[key])
        except Exception as e:
            continue


    print('CodeBLEU:', np.mean(bleu['codebleu']))
    print('Exact Match Score:', np.mean(bleu['codebleu'][bleu['codebleu']==1.0]))
    print('N-gram Match Score:', np.mean(bleu['ngram_match_score']))
    print('Weighted N-gram Match Score:', np.mean(bleu['weighted_ngram_match_score']))
    print('Syntax Match Score:', np.mean(bleu['syntax_match_score']))
    print('Dataflow Match Score:', np.mean(bleu['dataflow_match_score']))
    print('-'*50)