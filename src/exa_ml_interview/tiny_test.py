import torch
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizerFast, BertModel
from tqdm import tqdm
import json
import os
import torch.nn as nn
import torch.nn.functional as F
import random
from huggingface_hub import hf_hub_download

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORPUS_FRACTION = 0.01
NUM_QUERIES = 1000
RANDOM_SEED = 42
BATCH_SIZE = 32
REPO_ID = "charlieoneill/exa-int"
FILENAME = "biencoder_model.pth"

torch.set_grad_enabled(False)

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

def prepare_evaluation_data():
    print("Preparing evaluation data...")
    
    # Load datasets
    corpus = load_dataset("mteb/msmarco-v2", "corpus")['corpus']
    queries = load_dataset("mteb/msmarco-v2", "queries")['queries']
    default = load_dataset("mteb/msmarco-v2", "default", split="dev")

    # Sample corpus
    corpus_sample_size = int(len(corpus) * CORPUS_FRACTION)
    corpus_sample = corpus.shuffle(seed=RANDOM_SEED).select(range(corpus_sample_size))

    if os.path.exists('corpus_id_to_index.json'):
        print("Loading existing corpus ID to index mapping...")
        with open('corpus_id_to_index.json', 'r') as f:
            corpus_id_to_index = json.load(f)
    else:
        print("Creating new corpus ID to index mapping...")
        corpus_id_to_index = {item['_id']: idx for idx, item in enumerate(tqdm(corpus_sample))}
        with open('corpus_id_to_index.json', 'w') as f:
            json.dump(corpus_id_to_index, f)

    # Filter default to match sampled corpus
    default_filtered = default.filter(lambda x: x['corpus-id'] in corpus_id_to_index)

    # Select 1000 queries
    NUM_QUERIES = min(1000, len(default_filtered)) 
    selected_queries = default_filtered.shuffle(seed=RANDOM_SEED).select(range(NUM_QUERIES))
    
    # Create query dataset
    query_ids = set(selected_queries['query-id'])
    queries_filtered = queries.filter(lambda x: x['_id'] in query_ids)
    
    return corpus_sample, queries_filtered, selected_queries, corpus_id_to_index

@torch.no_grad()
def embed_texts(model, tokenizer, texts):
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs.pop('token_type_ids', None)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            batch_embeddings = model(**inputs).cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.concatenate(embeddings, axis=0)

def save_embeddings_and_mapping(embeddings, id_to_index, texts, filename_prefix):
    np.save(f"{filename_prefix}_embeddings.npy", embeddings)
    mapping = {doc_id: {"index": idx, "text": texts[idx]} for doc_id, idx in id_to_index.items()}
    with open(f"{filename_prefix}_mapping.json", "w") as f:
        json.dump(mapping, f)

def load_embeddings_and_mapping(filename_prefix):
    embeddings = np.load(f"{filename_prefix}_embeddings.npy")
    with open(f"{filename_prefix}_mapping.json", "r") as f:
        mapping = json.load(f)
    return embeddings, mapping

def calculate_recall_at_1(similarities, query_ids, corpus_ids, corpus_id_to_index):
    top_indices = similarities.argmax(axis=1)
    correct = 0
    for i, query_id in enumerate(query_ids):
        predicted_corpus_id = list(corpus_id_to_index.keys())[top_indices[i]]
        if predicted_corpus_id == corpus_ids[i]:
            correct += 1
    return correct / len(query_ids)

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    model = load_model("biencoder_model.pth")
    tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')

    if not os.path.exists("corpus_embeddings.npy") or not os.path.exists("corpus_mapping.json"):
        corpus_sample, queries_filtered, selected_queries, corpus_id_to_index = prepare_evaluation_data()
        
        print("Embedding corpus...")
        corpus_embeddings = embed_texts(model, tokenizer, corpus_sample['text'])
        save_embeddings_and_mapping(corpus_embeddings, corpus_id_to_index, corpus_sample['text'], "corpus")
        
        print("Saving query data...")
        with open("query_data.json", "w") as f:
            json.dump({
                "query_ids": selected_queries['query-id'],
                "corpus_ids": selected_queries['corpus-id'],
                "query_texts": queries_filtered['text']
            }, f)
    else:
        print("Loading pre-computed embeddings and mappings...")
        corpus_embeddings, corpus_mapping = load_embeddings_and_mapping("corpus")
        with open("query_data.json", "r") as f:
            query_data = json.load(f)

    print("Embedding queries...")
    query_embeddings = embed_texts(model, tokenizer, query_data['query_texts'])

    print("Calculating similarities...")
    similarities = np.dot(query_embeddings, corpus_embeddings.T)

    print("Calculating recall@1...")
    recall_at_1 = calculate_recall_at_1(similarities, query_data['query_ids'], query_data['corpus_ids'], corpus_mapping)

    print(f"Recall@1: {recall_at_1:.4f}")

if __name__ == "__main__":
    main()