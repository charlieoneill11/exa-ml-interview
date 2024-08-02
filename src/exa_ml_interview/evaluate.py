# import torch
# from transformers import BertTokenizerFast
# import numpy as np
# import json
# from datasets import load_dataset
# from dataclasses import dataclass
# from typing import List
# import time
# from tqdm import tqdm

# @dataclass
# class Result:
#     document_id: str
#     document_text: str
#     score: float

# class Retriever:
#     def __init__(self, model_path, embeddings_path, id_to_index_path, corpus_path):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = self.load_model(model_path)
#         self.tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')
#         self.corpus_embeddings = np.load(embeddings_path)
#         with open(id_to_index_path, 'r') as f:
#             self.id_to_index = json.load(f)
#         self.index_to_id = {v: k for k, v in self.id_to_index.items()}
#         self.corpus = load_dataset(corpus_path, split="corpus")

#     def load_model(self, model_path):
#         model = torch.load(model_path, map_location=self.device)
#         model.eval()
#         return model

#     def search(self, query: str, k: int) -> List[Result]:
#         # Tokenize and embed the query
#         inputs = self.tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(self.device)
#         with torch.no_grad():
#             query_embedding = self.model(**inputs).cpu().numpy()

#         # Compute similarities
#         similarities = np.dot(self.corpus_embeddings, query_embedding.T).squeeze()

#         # Get top k results
#         top_k_indices = similarities.argsort()[-k:][::-1]
#         results = []
#         for idx in top_k_indices:
#             doc_id = self.index_to_id[str(idx)]
#             doc_text = self.corpus[int(idx)]['text']
#             score = similarities[idx]
#             results.append(Result(document_id=doc_id, document_text=doc_text, score=float(score)))

#         return results

# def evaluate_retriever(retriever: Retriever, num_queries: int = 1000):
#     # Load dev queries
#     dev_queries = load_dataset("mteb/msmarco-v2", "queries", split="dev")
#     dev_queries = dev_queries.shuffle(seed=42).select(range(num_queries))

#     correct = 0
#     total_time = 0
#     slow_queries = 0

#     for query in tqdm(dev_queries, desc="Evaluating queries"):
#         start_time = time.time()
#         results = retriever.search(query['text'], k=1)
#         end_time = time.time()

#         search_time = end_time - start_time
#         total_time += search_time
#         if search_time > 1:
#             slow_queries += 1

#         # Check if the top result is correct (assuming the first result is the correct one)
#         if results[0].document_id == query['corpus-id']:
#             correct += 1

#     recall_at_1 = correct / num_queries
#     avg_search_time = total_time / num_queries
#     percent_slow = (slow_queries / num_queries) * 100

#     print(f"Recall@1: {recall_at_1:.4f}")
#     print(f"Average search time: {avg_search_time:.4f} seconds")
#     print(f"Percentage of queries taking >1 second: {percent_slow:.2f}%")

#     return recall_at_1, avg_search_time, percent_slow

# if __name__ == "__main__":
#     # Initialize the retriever with the paths to your model and data
#     retriever = Retriever(
#         model_path="biencoder_model.pth",
#         embeddings_path="corpus_embeddings.npy",
#         id_to_index_path="corpus_id_to_index.json",
#         corpus_path="mteb/msmarco-v2"
#     )

#     # Evaluate the retriever
#     recall, avg_time, slow_percent = evaluate_retriever(retriever)

#     # You can add additional code here to save the results or perform further analysis

import torch
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizerFast
from tqdm import tqdm
import json
import os
import random

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORPUS_FRACTION = 0.01
NUM_QUERIES = 1000
BATCH_SIZE = 32
RANDOM_SEED = 42

def load_model(model_path):
    # Use the BiEncoder class definition from the previous code
    model = BiEncoder().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
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
    corpus_id_to_index = {item['_id']: idx for idx, item in enumerate(corpus_sample)}

    # Filter default to match sampled corpus
    default_filtered = default.filter(lambda x: x['corpus-id'] in corpus_id_to_index)

    # Select 1000 queries
    selected_queries = default_filtered.shuffle(seed=RANDOM_SEED).select(range(NUM_QUERIES))
    
    # Create query dataset
    query_ids = set(selected_queries['query-id'])
    queries_filtered = queries.filter(lambda x: x['_id'] in query_ids)
    
    return corpus_sample, queries_filtered, selected_queries, corpus_id_to_index

def embed_texts(model, tokenizer, texts):
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
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