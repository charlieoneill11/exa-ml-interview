from datasets import load_dataset
import json
import random

# load datasets
corpus = load_dataset("mteb/msmarco-v2", "corpus")
queries = load_dataset("mteb/msmarco-v2", "queries")
default = load_dataset("mteb/msmarco-v2", "default")

dev_set = default['dev']

query_dict = {q['_id']: q['text'] for q in queries['queries']}

eval_queries = []

used_query_ids = set()

while len(eval_queries) < 1000:
    idx = random.randint(0, len(dev_set) - 1)
    query_id = dev_set[idx]['query-id']
    
    if query_id not in used_query_ids and query_id in query_dict:
        used_query_ids.add(query_id)
        eval_queries.append({
            'query_id': query_id,
            'query_text': query_dict[query_id],
            'relevant_doc_id': dev_set[idx]['corpus-id']
        })

with open('eval_queries.json', 'w') as f:
    json.dump(eval_queries, f)

print(f"Generated and saved {len(eval_queries)} evaluation queries to eval_queries.json")