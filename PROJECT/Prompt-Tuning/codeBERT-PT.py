from datasets import load_dataset
def langdata(lang):
    from unittest.mock import patch
    def mock_input(prompt):
        print(prompt)
        return "y"

    with patch('builtins.input', mock_input):

        dataset = load_dataset("code_search_net", lang)
    return dataset


maskdata=[]
for _ in ['python','javascript','java','ruby']:
    dataset = langdata(_)
    print("$"*50)
    for i in dataset['train'].select(range(500)):
        maskdata.append(i['func_code_string'])

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForMaskedLM, AdamW
import random
from tqdm import tqdm
import os

model_name = "microsoft/codebert-base-mlm"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(model_name)

num_prompt_tokens = 10

vocab_size, embed_dim = model.roberta.embeddings.word_embeddings.weight.shape
prompt_embeddings = torch.nn.Parameter(torch.randn(num_prompt_tokens, embed_dim))

for param in model.parameters():
    param.requires_grad = False

def model_forward(input_ids, attention_mask, labels=None):
    device = input_ids.device
    prompt_embeddings_device = prompt_embeddings.to(device)
    
    inputs_embeds = model.roberta.embeddings.word_embeddings(input_ids[:, num_prompt_tokens:])
    
    inputs_embeds = torch.cat([prompt_embeddings_device.unsqueeze(0).expand(input_ids.shape[0], -1, -1), inputs_embeds], dim=1)
    
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
    return outputs

class CodeDataset(Dataset):
    def __init__(self, code_list, tokenizer, max_length=512, mask_probability=0.15):
        self.code_list = code_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability

    def __len__(self):
        return len(self.code_list)

    def __getitem__(self, idx):
        code = self.code_list[idx]
        
        inputs = self.tokenizer(code, truncation=True, max_length=self.max_length-num_prompt_tokens, padding="max_length", return_tensors="pt")
        
        input_ids = torch.cat([torch.full((1, num_prompt_tokens), self.tokenizer.pad_token_id), inputs["input_ids"]], dim=1)
        attention_mask = torch.cat([torch.ones(1, num_prompt_tokens), inputs["attention_mask"]], dim=1)
        

        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < self.mask_probability) * (input_ids != self.tokenizer.pad_token_id) * (input_ids != self.tokenizer.cls_token_id) * (input_ids != self.tokenizer.sep_token_id)
        selection = torch.flatten(mask_arr.nonzero()).tolist()
        input_ids[0, selection] = self.tokenizer.mask_token_id

        return {
            "input_ids": input_ids.squeeze(),
            "attention_mask": attention_mask.squeeze(),
            "labels": input_ids.clone().squeeze()
        }

def save_model(model, prompt_embeddings, tokenizer, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.save_pretrained(save_dir)
    
    torch.save(prompt_embeddings, os.path.join(save_dir, "prompt_embeddings.pt"))
    
    tokenizer.save_pretrained(save_dir)
    print("hello")
def train(train_dataloader, num_epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prompt_embeddings.to(device)
    optimizer = AdamW([prompt_embeddings], lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model_forward(input_ids, attention_mask, labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        save_model(model,prompt_embeddings,tokenizer,f"cpu-save_dir/{epoch}")
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader)}")

def predict_masked_tokens(input_text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prompt_embeddings.to(device)
    
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = torch.cat([torch.full((1, num_prompt_tokens), tokenizer.pad_token_id), inputs["input_ids"]], dim=1).to(device)
    attention_mask = torch.cat([torch.ones(1, num_prompt_tokens), inputs["attention_mask"]], dim=1).to(device)
    
    mask_token_indices = torch.where(input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        outputs = model_forward(input_ids, attention_mask)
    
    logits = outputs.logits
    predictions = {}
    for idx in mask_token_indices:
        masked_token_logits = logits[0, idx, :]
        top_5_tokens = torch.topk(masked_token_logits, 5, dim=0).indices.tolist()
        predicted_tokens = tokenizer.convert_ids_to_tokens(top_5_tokens)
        predictions[idx.item()] = predicted_tokens
    
    return predictions

def train_codebert_with_prompt_tuning(python_programs, num_epochs=50, batch_size=2, learning_rate=1e-3):
    dataset = CodeDataset(python_programs, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    train(dataloader, num_epochs, learning_rate)
    
    return model, tokenizer


trained_model, trained_tokenizer = train_codebert_with_prompt_tuning(maskdata)

masked_text = "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    else:\n        pivot = arr[<mask>]\n        left = [x for x in arr[1:] if x <mask> pivot]\n        right = [x for x in arr[1:] if x >= pivot]\n        return <mask>(left) + [pivot] + quicksort(right)"
predictions = predict_masked_tokens(masked_text)
for idx, tokens in predictions.items():
    print(f"Predictions for mask at position {idx}: {tokens}")
