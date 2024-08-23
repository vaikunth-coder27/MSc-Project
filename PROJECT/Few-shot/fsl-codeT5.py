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
import random
from codebleu import calc_codebleu
import numpy as np
from tqdm import tqdm
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

warnings.filterwarnings("ignore")
logging.getLogger('transformers').setLevel(logging.ERROR)

class dummy_args:
    def __init__(self):
        self.prog_lang = 'java'
        self.device = 'cuda'
        self.model_name = 'Salesforce/codet5-large'
        self.quantization = False
        self.number_of_shot = 3
        self.number_of_examples = 10

args = dummy_args()

# parser = argparse.ArgumentParser(description='Extracting memorized content from code snippets')
# parser.add_argument('--prog_lang', type=str, default='python', help='Programming language of the code snippets',choices=['python','java','javascript','ruby'])
# parser.add_argument('--device', type=str, default='cpu', help='Device to run the extraction process on', choices=['cpu','cuda'])
# parser.add_argument('--model_name', type=str, default='Salesforce/codet5-large', help='Name of the autoregressive model to use for extracting content')
# parser.add_argument('--quantization', type=bool, default=False, help='Whether to quantize the model or not')
# args = parser.parse_args()

print(f'Using device: {args.device}')
print(f'Programming language: {args.prog_lang}')
print(f'Model name: {args.model_name}')
print(f'Quantization: {args.quantization}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prog_language = args.prog_lang
dataset = load_dataset("code_search_net", prog_language)


T5_PATH = args.model_name

t5_tokenizer = transformers.AutoTokenizer.from_pretrained(T5_PATH)
t5_config = transformers.T5Config.from_pretrained(T5_PATH)
t5_mlm = transformers.AutoModelForSeq2SeqLM.from_pretrained(T5_PATH, config=t5_config).to(device)
if args.quantization:
    t5_mlm = torch.quantization.quantize_dynamic(t5_mlm, {torch.nn.Linear}, dtype=torch.qint8)
    print('Model quantized successfully')

print('Model, dataset and tokenizer loaded successfully')


def t5_generate_code_suffix(prefix,model,tokenizer, max_new_tokens=1024):
    input_ids = tokenizer(prefix, return_tensors="pt").to(device).input_ids
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_code



print('initiating extraction process...')

#zeroshot_bleu_t5={'codebleu':np.array([]),'ngram_match_score':np.array([]),'weighted_ngram_match_score':np.array([]),'syntax_match_score':np.array([]),'dataflow_match_score':np.array([])}
oneshot_bleu_t5={'codebleu':np.array([]),'ngram_match_score':np.array([]),'weighted_ngram_match_score':np.array([]),'syntax_match_score':np.array([]),'dataflow_match_score':np.array([])}
twoshot_bleu_t5={'codebleu':np.array([]),'ngram_match_score':np.array([]),'weighted_ngram_match_score':np.array([]),'syntax_match_score':np.array([]),'dataflow_match_score':np.array([])}
threeshot_bleu_t5={'codebleu':np.array([]),'ngram_match_score':np.array([]),'weighted_ngram_match_score':np.array([]),'syntax_match_score':np.array([]),'dataflow_match_score':np.array([])}




for i in range(100):
    #random_numbers = [random.randint(0, 400000) for _ in range(100)]
    print(f"Extracting memorized content for samples {100*i} to {(i+1)*100}...")
    test = random.randint(0, 4000)
    for instance in tqdm(dataset['train'].select(range(100*i,100*(i+1)))):#random_numbers)):
        code = instance['func_code_string']
        temp = code.split('\n')
        prefix_code = '\n'.join(temp[:len(temp)//2])+ '\n'
        suffix_code = '\n'.join(temp[len(temp)//2:])
        j=0

        # prompt_code =dataset['train'][random.randint(0, 400000)]['func_code_string']
        # second_prompt_code =dataset['train'][random.randint(0, 400000)]['func_code_string']
        # third_prompt_code =dataset['train'][random.randint(0, 400000)]['func_code_string']
        # prompt_1shot = 'Prefix: '+prompt_code[:len(prompt_code)//2]+'\n'+ 'Suffix: '+prompt_code[len(prompt_code)//2:]+'\n'+'prefix: '+second_prompt_code[:len(second_prompt_code)//2]+'\n'+ 'Suffix: '+second_prompt_code[len(second_prompt_code)//2:]+'\n'+ 'prefix: '+third_prompt_code[:len(third_prompt_code)//2]+'\n'+ 'Suffix: '+third_prompt_code[len(third_prompt_code)//2:]+'\n'+ 'Prefix: '+prefix_code+'\n'
        while j<args.number_of_examples:
            prompt_1shot=''
            for _ in range(args.number_of_shot):
                prompt_code =dataset['train'][random.randint(0, 400000)]['func_code_string']
                prompt_1shot+='Prefix: '+prompt_code[:len(prompt_code)//2]+'\n'+ 'Suffix: '+prompt_code[len(prompt_code)//2:]+'\n'
            prompt_1shot+='Prefix: '+prefix_code+'\n'+ 'Suffix: '
            try:
                #zeroshot_t5_top_tokens = t5_generate_code_suffix(prefix_code,t5_mlm,t5_tokenizer)
                oneshot_t5_top_tokens = t5_generate_code_suffix(prompt_1shot,t5_mlm,t5_tokenizer)

                # print('Prefix:', prefix_code)
                # print('Suffix:', suffix_code)
                # print('Generated:', t5_top_tokens)
                #print('-'*50)
                #bleu1 = calc_codebleu([suffix_code],[zeroshot_t5_top_tokens],lang=prog_language, weights=(0.25, 0.25, 0.25, 0.25))
                bleu2 = calc_codebleu([suffix_code],[oneshot_t5_top_tokens],lang=prog_language, weights=(0.25, 0.25, 0.25, 0.25))
                print(bleu2)
                j+=1
            except Exception as e:
                continue
            for key in oneshot_bleu_t5.keys():
                    #zeroshot_bleu_t5[key] = np.append(zeroshot_bleu_t5[key],bleu2[key])
                    oneshot_bleu_t5[key] = np.append(oneshot_bleu_t5[key],bleu2[key])




    #print('Zero-shot CodeBLEU:', np.mean(zeroshot_bleu_t5['codebleu']))
    print('One-shot CodeBLEU:', np.mean(oneshot_bleu_t5['codebleu']))
    print('#'*50)



    # print('-'*50)
    # break
