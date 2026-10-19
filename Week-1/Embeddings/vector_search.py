import numpy as np
from sentence_transformers import SentenceTransformer

DOCS = [
    "A 429 status code means you sent too many requests and hit a rate limit.",
    "A 401 status code means the API key is missing or invalid.",
    "Never commit your .env file; add it to .gitignore so your API key stays private.",
    "json.loads converts a JSON string into a Python dictionary.",
    "Temperature controls how random the model output is; low values give repeatable answers.",
    "Embeddings turn text into vectors so that similar meanings sit close together.",
    "Cosine similarity measures the angle between two vectors.",
    "A virtual environment keeps each project's libraries separate.",
    "RAG retrieves relevant documents and gives them to the LLM to answer questions.",
    "Overfitting happens when a model memorizes training data and fails on new data.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_emb = model.encode(DOCS, normalize_embeddings=True)   # shape: (10, 384)


def search(query, k=3):
    q = model.encode(query, normalize_embeddings=True)    # shape: (384,)
    scores = doc_emb @ q                                  # one score per document
    top = np.argsort(-scores)[:k]                         # indices of the k best
    return [(DOCS[i], float(scores[i])) for i in top]


MIN_SCORE = 0.3   # tune on your own data


def search(query, k=3, min_score=MIN_SCORE):
    q = model.encode(query, normalize_embeddings=True)
    scores = doc_emb @ q
    top = np.argsort(-scores)[:k]
    return [(DOCS[i], float(scores[i])) for i in top if scores[i] >= min_score]


def main():
    print("Ask a question (type 'quit' to exit)")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not query:                      # user just pressed Enter
            continue

        results = search(query)
        if not results:
            print("No relevant documents found.")
            continue
        for text, score in results:
            print(f"  {score:.2f}  {text}")


if __name__ == "__main__":
    main()