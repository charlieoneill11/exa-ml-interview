import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import json
import os
import random
import nltk
from nltk.corpus import stopwords
import argparse

# Constants
CORPUS_FRACTION = 0.001
NUM_QUERIES = 1000
RANDOM_SEED = 42

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize and remove stopwords
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join([token for token in tokens if token not in stop_words])

def prepare_evaluation_data():
    print("Preparing evaluation data...")
    
    corpus = load_dataset("mteb/msmarco-v2", "corpus")['corpus']
    queries = load_dataset("mteb/msmarco-v2", "queries")['queries']
    default = load_dataset("mteb/msmarco-v2", "default", split="dev")

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

    # filter default to match sampled corpus
    default_filtered = default.filter(lambda x: x['corpus-id'] in corpus_id_to_index)

    # select queries
    NUM_QUERIES = min(1000, len(default_filtered))
    selected_queries = default_filtered.shuffle(seed=RANDOM_SEED).select(range(NUM_QUERIES))
    
    query_ids = set(selected_queries['query-id'])
    queries_filtered = queries.filter(lambda x: x['_id'] in query_ids)
    
    return corpus_sample, queries_filtered, selected_queries, corpus_id_to_index

def calculate_recall_at_n(similarities, query_ids, corpus_ids, corpus_id_to_index, n):
    top_indices = np.argsort(-similarities, axis=1)[:, :n]
    correct = 0
    for i, query_id in enumerate(query_ids):
        predicted_corpus_ids = [list(corpus_id_to_index.keys())[idx] for idx in top_indices[i]]
        if corpus_ids[i] in predicted_corpus_ids:
            correct += 1
    return correct / len(query_ids)

def main(n):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    corpus_sample, queries_filtered, selected_queries, corpus_id_to_index = prepare_evaluation_data()

    print("Preprocessing corpus...")
    texts = corpus_sample['text']
    print(f"Corpus size: {len(texts)}")
    corpus_texts = [preprocess_text(text) for text in tqdm(texts)]

    print("Calculating TF-IDF for corpus...")
    vectorizer = TfidfVectorizer()
    corpus_tfidf = vectorizer.fit_transform(corpus_texts)

    print("Preprocessing queries...")
    query_texts = [preprocess_text(text) for text in tqdm(queries_filtered['text'])]

    print("Calculating TF-IDF for queries...")
    query_tfidf = vectorizer.transform(query_texts)

    print("Calculating similarities...")
    similarities = cosine_similarity(query_tfidf, corpus_tfidf)

    print(f"Calculating recall@{n}...")
    recall_at_n = calculate_recall_at_n(similarities, selected_queries['query-id'], selected_queries['corpus-id'], corpus_id_to_index, n)

    print(f"Recall@{n}: {recall_at_n:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Recall@N for TF-IDF based retrieval.')
    parser.add_argument('--n', type=int, default=1, help='N value for Recall@N calculation')
    args = parser.parse_args()
    
    main(args.n)