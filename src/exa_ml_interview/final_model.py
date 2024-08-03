### NOTE: this is untested as embeddings were still running at the end of day ###

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
from dataclasses import dataclass
from typing import List

BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ID = "charlieoneill/exa-int"
FILENAME = "biencoder_model.pth"

@dataclass
class Result:
    document_id: str
    document_text: str
    score: float

class BiEncoder(nn.Module):
    def __init__(self):
        super(BiEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-large-uncased')
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(outputs.pooler_output, p=2, dim=-1)

class Retriever:
    def __init__(self):
        self.model = self.load_model("biencoder_model.pth")
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')
        self.corpus = load_dataset("mteb/msmarco-v2", "corpus")['corpus']
        self.embeddings = np.load("corpus_embeddings.npy")
        with open("corpus_id_to_index.json", "r") as f:
            self.id_to_index = json.load(f)

    def load_model(self, model_path):
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
                model = state_dict
            else:
                model.bert = BertModel.from_pretrained('bert-large-uncased')
                model.bert.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Initializing a new model instead.")
            model.bert = BertModel.from_pretrained('bert-large-uncased')
        
        model.eval()
        return model

    def embed_query(self, query):
        inputs = self.tokenizer([query], padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs.pop('token_type_ids', None)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            query_embedding = self.model(**inputs).cpu().numpy()
        
        return query_embedding

    def search(self, query: str, k: int) -> List[Result]:
        query_embedding = self.embed_query(query)
        scores = np.dot(self.embeddings, query_embedding.T).squeeze()
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            doc_id = list(self.id_to_index.keys())[list(self.id_to_index.values()).index(idx)]
            doc_text = self.corpus[idx]['text']
            score = float(scores[idx])
            results.append(Result(document_id=doc_id, document_text=doc_text, score=score))
        
        return results

def embed_corpus(retriever):
    print("Embedding corpus...")
    embeddings = []
    id_to_index = {}
    
    for i in tqdm(range(0, len(retriever.corpus), BATCH_SIZE), desc="Embedding corpus"):
        batch = retriever.corpus[i:i+BATCH_SIZE]
        texts = batch['text']
        ids = batch['_id']
        
        inputs = retriever.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs.pop('token_type_ids', None)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            batch_embeddings = retriever.model(**inputs).cpu().numpy()
        
        embeddings.append(batch_embeddings)
        for idx, doc_id in enumerate(ids):
            id_to_index[doc_id] = i + idx
    
    all_embeddings = np.concatenate(embeddings, axis=0)
    
    print("Saving embeddings and id_to_index mapping...")
    np.save("corpus_embeddings.npy", all_embeddings)
    with open("corpus_id_to_index.json", "w") as f:
        json.dump(id_to_index, f)
    
    print(f"Saved {all_embeddings.shape[0]} embeddings of dimension {all_embeddings.shape[1]}")
    print(f"Saved id_to_index mapping for {len(id_to_index)} documents")

if __name__ == "__main__":
    retriever = Retriever()
    embed_corpus(retriever)