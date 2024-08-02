import torch
from transformers import BertTokenizerFast, BertModel
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import numpy as np
import json
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import os

# Constants
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ID = "charlieoneill/exa-int"
FILENAME = "biencoder_model.pth"

class BiEncoder(nn.Module):
    def __init__(self):
        super(BiEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-large-uncased')
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(outputs.pooler_output, p=2, dim=-1)

def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model not found locally. Downloading from Hugging Face Hub: {REPO_ID}")
        try:
            model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, token="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")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Initializing a new model instead.")
            model = BiEncoder().to(DEVICE)
            model.bert = BertModel.from_pretrained('bert-large-uncased')
            return model

    model = BiEncoder().to(DEVICE)
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        if isinstance(state_dict, BiEncoder):
            # If the saved object is the entire model
            model = state_dict
        else:
            # If the saved object is just the state dict
            model.bert = BertModel.from_pretrained('bert-large-uncased')
            model.bert.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Initializing a new model instead.")
        model.bert = BertModel.from_pretrained('bert-large-uncased')
    
    model.eval()
    return model

def embed_corpus():
    # Load the model and tokenizer
    model = load_model("biencoder_model.pth")
    tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')
    
    # Load the corpus
    print("Loading corpus...")
    corpus = load_dataset("mteb/msmarco-v2", "corpus")['corpus']
    
    # Initialize embeddings array and id_to_index mapping
    embeddings = []
    id_to_index = {}
    
    # Process corpus in batches
    for i in tqdm(range(0, len(corpus), BATCH_SIZE), desc="Embedding corpus"):
        batch = corpus[i:i+BATCH_SIZE]
        print(batch)
        texts = batch['text'] #[item['text'] for item in batch]
        ids = batch['_id'] #[item['_id'] for item in batch]
        
        # Tokenize
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        # Remove token_type_ids if present
        inputs.pop('token_type_ids', None)
        
         # Move inputs to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            batch_embeddings = model(**inputs).cpu().numpy()
        
        # Store embeddings and update id_to_index mapping
        embeddings.append(batch_embeddings)
        for idx, doc_id in enumerate(ids):
            id_to_index[doc_id] = i + idx
    
    # Concatenate all embeddings
    all_embeddings = np.concatenate(embeddings, axis=0)
    
    # Save embeddings and id_to_index mapping
    print("Saving embeddings and id_to_index mapping...")
    np.save("corpus_embeddings.npy", all_embeddings)
    with open("corpus_id_to_index.json", "w") as f:
        json.dump(id_to_index, f)
    
    print(f"Saved {all_embeddings.shape[0]} embeddings of dimension {all_embeddings.shape[1]}")
    print(f"Saved id_to_index mapping for {len(id_to_index)} documents")

if __name__ == "__main__":
    embed_corpus()