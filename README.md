# Exa ML Interview

This is the starter repo for the onsight Exa ML interview! You will fork this and work on it yourself.

Your goal is to train a model/system for document retrieval over the MS Marco dataset. In particular, at the end of this you will implement a class `Retriever`

```python
@dataclass
class Result:
    document_id: str
    document_text: str
    score: float

class Retriever:
    def __init__(self):
        pass

    def search(query: str, k: int) -> List[Result]:
        pass
```

After the interview, I will call search() on 1000 queries from the MS-Marco test set, and compute the recall@1. Your goal is to produce a system that has the highest recall@1. 

You are welcome to use any approach you want for this problem -- it will likely involve a transformer-backed bi-encoder at some level, but are welcome to use any architecture or other system to improve the performance. You do have the following constraints:

1. search must return a result in <1 second (95% of the time)
2. You will have access to a g5.4xlarge aws instance to do all the training on
3. You are welcome to used pretrained LLMs in any capacity here, but you can't start with a pretrained embedding-specific model
4. You should only train on the MS-Marco train set -- or any synthetic data you generate

## Evaluation

What we're looking for in this interview is

1) Well written, readable, working code
2) That gets good accuracy on the evaluation
3) That indicates good ML research -- you should try out approaches, from the literature or your own ideas, and validate they work

In particular, I recommend getting a simple bi-encoder retriever training first, and once you have that running then try out novel improvements, either to the retrieval architecture, the data, the training code, or the model code. We are looking for a system that gets good accuracy, but beyond that we're really looking for evidence that you can do good applied ML research -- we're interested in seeing what experiments and novel solutions you come up with!

Feel free to ask questions! This is meant to be a collaborative exercise.

