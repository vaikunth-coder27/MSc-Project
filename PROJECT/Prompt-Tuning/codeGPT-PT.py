from datasets import load_dataset
def langdata(lang):
    from unittest.mock import patch
    def mock_input(prompt):
        print(prompt)
        return "y"

    with patch('builtins.input', mock_input):

        dataset = load_dataset("code_search_net", lang)
    return dataset


prefix_suffix_pairs=[]
for _ in ['python','javascript','java','ruby']:
    dataset = langdata(_)
    for i in dataset['train'].select(range(500)):
        prefix_suffix_pairs.append((i['func_code_string'][:len(i['func_code_string'])//2],i['func_code_string'][len(i['func_code_string'])//2:]))


import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

class CodeCompletionDataset(Dataset):
    def __init__(self, prefix_suffix_pairs, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.prefix_suffix_pairs = prefix_suffix_pairs
        self.max_length = max_length

    def __len__(self):
        return len(self.prefix_suffix_pairs)

    def __getitem__(self, idx):
        prefix, suffix = self.prefix_suffix_pairs[idx]
        full_text = prefix + suffix
        inputs = self.tokenizer.encode_plus(full_text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        return {'input_ids': inputs['input_ids'].squeeze(), 'attention_mask': inputs['attention_mask'].squeeze()}

class PromptTunedCodeGPT(torch.nn.Module):
    def __init__(self, model_name, num_virtual_tokens=10):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.num_virtual_tokens = num_virtual_tokens
        self.virtual_tokens = torch.nn.Parameter(torch.randn(num_virtual_tokens, self.model.config.n_embd))

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        virtual_tokens = self.virtual_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
        
        inputs_embeds = self.model.transformer.wte(input_ids)
        inputs_embeds = torch.cat([virtual_tokens, inputs_embeds], dim=1)
        
        attention_mask = torch.cat([torch.ones(batch_size, self.num_virtual_tokens, device=attention_mask.device), attention_mask], dim=1)
        
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    
    def generate(self, input_ids, attention_mask, **kwargs):
        batch_size = input_ids.shape[0]
        virtual_tokens = self.virtual_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
        
        inputs_embeds = self.model.transformer.wte(input_ids)
        inputs_embeds = torch.cat([virtual_tokens, inputs_embeds], dim=1)
        
        attention_mask = torch.cat([torch.ones(batch_size, self.num_virtual_tokens, device=attention_mask.device), attention_mask], dim=1)
        
        return self.model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        shift_logits = logits[:, model.num_virtual_tokens:-1, :].contiguous()
        
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def save_prompt_tuned_model(model, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    model.model.config.save_pretrained(save_directory)
    model.tokenizer.save_pretrained(save_directory)
    torch.save(model.virtual_tokens, os.path.join(save_directory, "virtual_tokens.pt"))
    print(f"Model saved to {save_directory}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'AISE-TUDelft/CodeGPT-Multilingual'  # You can change this to other CodeGPT variants

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
dataset = CodeCompletionDataset(prefix_suffix_pairs, tokenizer, max_length=512)
print('Dataset loading done')
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
print('Dataloader loading done')

model = PromptTunedCodeGPT(model_name).to(device)
print('Model loading done')
optimizer = AdamW([model.virtual_tokens], lr=1e-3)

save_dir = "cpu-CodeGPT-PT"
num_epochs = 50
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    loss = train(model, dataloader, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch + 1}")
    save_prompt_tuned_model(model, epoch_save_dir)
    print(f"Model saved after epoch {epoch + 1}")


def load_prompt_tuned_model(model_name, save_directory):
    base_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(save_directory)
    
    model = PromptTunedCodeGPT(model_name)
    
    model.virtual_tokens.data = torch.load(os.path.join(save_directory, "virtual_tokens.pt"))
    
    return model, tokenizer

t5_mlm,t5_tokenizer = load_prompt_tuned_model("AISE-TUDelft/CodeGPT-Multilingual", "cpu-CodeGPT-PT/epoch_49")
print("<<<<<TRAINING COMPLETE>>>>>>>")
