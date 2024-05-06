from datasets import load_dataset
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer
from tqdm import tqdm

class TurkishSummaryDataset(Dataset):
    def __init__(self, data, tokenizer, src_column, dest_column, seq_length=512):
        self.dataset = data
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.src_column = src_column
        self.dest_column = dest_column
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        full_text = self.dataset[self.src_column][index]
        summary = self.dataset[self.dest_column][index]
        
        input_encoding = self.tokenizer(
            full_text,
            max_length=self.seq_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids = input_encoding['input_ids'].flatten()
        attention_mask = input_encoding['attention_mask'].flatten()
        
        target_encoding = self.tokenizer(
            summary,
            max_length=self.seq_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        target_ids = target_encoding['input_ids'].flatten()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_ids': target_ids
        }
            
def main():
    dataset = load_dataset("mlsum", "tu")
    model = GPT2LMHeadModel.from_pretrained("ytu-ce-cosmos/turkish-gpt2")
    tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2")
    data_size = 100
    
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = dataset['train'][:data_size]
    
    turkish_summary_dataset = TurkishSummaryDataset(dataset, tokenizer, 'text', 'summary')
    
    train_dataloader = DataLoader(
        turkish_summary_dataset,
        batch_size=1,
        shuffle=True
    )
                            
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=5e-7)
    model.train()
    for epoch in range(1):
        print('Epoch', epoch)

        train_losses = []
        print('Training...')

        progress_bar = tqdm(train_dataloader)
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(model.device)
            labels = batch['target_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            progress_bar.set_description("Loss: %.3f" % (sum(train_losses) / len(train_losses)))

    os.makedirs('./model', exist_ok=True)
    torch.save(model.state_dict(), './model/model1.pth')
    print("Fine-Tuning Completed")

if __name__ == "__main__":
    main()
