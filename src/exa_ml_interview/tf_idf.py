from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datasets import load_dataset
import random
from tqdm import tqdm
import time
import os
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TFIDFRetriever:
    def __init__(self, corpus, index_path='tfidf_index.joblib'):
        self.corpus = corpus
        self.index_path = index_path
        self.vectorizer = None
        self.tfidf_matrix = None
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        # Tokenize the text
        tokens = word_tokenize(text.lower())
        # Remove stop words and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        return ' '.join(tokens)

    def build_or_load_index(self):
        if os.path.exists(self.index_path):
            print(f"Loading existing TF-IDF index from {self.index_path}")
            loaded_data = joblib.load(self.index_path)
            self.vectorizer = loaded_data['vectorizer']
            self.tfidf_matrix = loaded_data['tfidf_matrix']
        else:
            print("Building new TF-IDF index...")
            self.vectorizer = TfidfVectorizer(preprocessor=self.preprocess_text)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus['text'])
            
            print(f"Saving TF-IDF index to {self.index_path}")
            joblib.dump({
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, self.index_path)

    def search(self, query, k=10):
        preprocessed_query = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([preprocessed_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'document_id': self.corpus[idx]['_id'],
                'document_text': self.corpus[idx]['text'],
                'score': similarities[idx]
            })
        return results

def main():
    print("Loading datasets...")
    corpus = load_dataset("mteb/msmarco-v2", "corpus", split="corpus")
    queries = load_dataset("mteb/msmarco-v2", "queries", split="queries")

    retriever = TFIDFRetriever(corpus)
    retriever.build_or_load_index()

    print("Selecting 10 random queries...")
    random_queries = random.sample(list(queries), 10)

    print("Testing TF-IDF search...")
    for query in tqdm(random_queries):
        start_time = time.time()
        results = retriever.search(query['text'], k=10)
        end_time = time.time()

        print(f"\nQuery: {query['text']}")
        print(f"Search time: {end_time - start_time:.4f} seconds")
        print("Top result:")
        print(f"Document ID: {results[0]['document_id']}")
        print(f"Score: {results[0]['score']:.4f}")
        print(f"Text snippet: {results[0]['document_text'][:200]}...")
        print("-" * 80)

if __name__ == "__main__":
    main()