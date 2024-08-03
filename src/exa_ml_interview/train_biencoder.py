import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizerFast, BertModel
import torch.nn.functional as F
import random
from tqdm import tqdm
import json
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

# Set device and random seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# Constants
BATCH_SIZE = 32
ACCURACY_INTERVAL = 10

class BiEncoder(nn.Module):
    def __init__(self):
        super(BiEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-large-uncased')
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(outputs.pooler_output, p=2, dim=-1)

def load_prepared_data(filename):
    print(f"Loading prepared data from {filename}...")
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples.")
    return data

def tokenize_batch(batch, tokenizer):
    queries = tokenizer([item['query'] for item in batch], padding=True, truncation=True, return_tensors='pt', max_length=512)
    documents = tokenizer([item['positive'] for item in batch], padding=True, truncation=True, return_tensors='pt', max_length=512)
    
    # Remove token_type_ids from all tokenized outputs
    for output in [queries, documents]:
        output.pop('token_type_ids', None)

    return {
        'queries': {k: v.to(device) for k, v in queries.items()},
        'documents': {k: v.to(device) for k, v in documents.items()}
    }

def in_batch_cross_entropy_loss(similarities):
    labels = torch.arange(similarities.size(0)).to(device)
    return nn.CrossEntropyLoss()(similarities, labels)

def recall_at_1(similarities):
    return (similarities.argmax(dim=1) == torch.arange(similarities.size(0)).to(device)).float().mean().item()

def evaluate(model, data, tokenizer):
    model.eval()
    losses = []
    recalls = []
    with torch.no_grad():
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i+BATCH_SIZE]
            tokenized_batch = tokenize_batch(batch, tokenizer)
            
            query_embeddings = model(**tokenized_batch['queries'])
            doc_embeddings = model(**tokenized_batch['documents'])
            
            similarities = torch.mm(query_embeddings, doc_embeddings.t())
            
            loss = in_batch_cross_entropy_loss(similarities)
            recall = recall_at_1(similarities)
            
            losses.append(loss.item())
            recalls.append(recall)
    
    return sum(losses) / len(losses), sum(recalls) / len(recalls)

def train_and_evaluate(resume_from=None):
    if resume_from and os.path.exists(resume_from):
        print(f"Loading model from {resume_from}")
        model = torch.load(resume_from)
    else:
        print("Initializing new model")
        model = BiEncoder()
    
    model = model.to(device)
    tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    train_data = load_prepared_data('msmarco_train_data.json')
    dev_data = load_prepared_data('msmarco_dev_data.json')
    
    train_losses = []
    train_recalls = []
    dev_losses = []
    dev_recalls = []
    
    for epoch in range(3):  # 3 epochs
        model.train()
        random.shuffle(train_data)  # Shuffle data at the start of each epoch
        for i in tqdm(range(0, len(train_data), BATCH_SIZE), desc=f"Epoch {epoch+1}"):
            batch = train_data[i:i+BATCH_SIZE]
            tokenized_batch = tokenize_batch(batch, tokenizer)
            
            optimizer.zero_grad()
            
            query_embeddings = model(**tokenized_batch['queries'])
            doc_embeddings = model(**tokenized_batch['documents'])
            
            similarities = torch.mm(query_embeddings, doc_embeddings.t())
            
            loss = in_batch_cross_entropy_loss(similarities)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_recalls.append(recall_at_1(similarities))
            
            if i % (ACCURACY_INTERVAL * BATCH_SIZE) == 0:
                avg_train_loss = sum(train_losses[-ACCURACY_INTERVAL:]) / min(ACCURACY_INTERVAL, len(train_losses))
                avg_train_recall = sum(train_recalls[-ACCURACY_INTERVAL:]) / min(ACCURACY_INTERVAL, len(train_recalls))
                
                dev_loss, dev_recall = evaluate(model, dev_data, tokenizer)
                dev_losses.append(dev_loss)
                dev_recalls.append(dev_recall)
                
                print(f"Epoch {epoch+1}, Batch {i//BATCH_SIZE}")
                print(f"Train - Loss: {avg_train_loss:.4f}, Recall@1: {avg_train_recall:.4f}")
                print(f"Dev   - Loss: {dev_loss:.4f}, Recall@1: {dev_recall:.4f}")
                print("-" * 50)

                torch.save(model, 'biencoder_model.pth')
    
    print("Training completed.")
    print(f"Final Train - Loss: {sum(train_losses[-100:]) / 100:.4f}, Recall@1: {sum(train_recalls[-100:]) / 100:.4f}")
    print(f"Final Dev   - Loss: {dev_losses[-1]:.4f}, Recall@1: {dev_recalls[-1]:.4f}")
    
    return train_losses, train_recalls, dev_losses, dev_recalls, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiEncoder model")
    parser.add_argument("--resume", type=str, help="Path to model file to resume training from", default=None)
    args = parser.parse_args()

    train_losses, train_recalls, dev_losses, dev_recalls, model = train_and_evaluate(resume_from=args.resume)

    # Save model
    torch.save(model, 'biencoder_model.pth')

    # Save everything else
    with open('train_losses.json', 'w') as f:
        json.dump(train_losses, f)

    with open('train_recalls.json', 'w') as f:
        json.dump(train_recalls, f)

    with open('dev_losses.json', 'w') as f:
        json.dump(dev_losses, f)

    with open('dev_recalls.json', 'w') as f:
        json.dump(dev_recalls, f)