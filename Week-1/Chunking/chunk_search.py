from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
DB_DIR = str(Path(__file__).parent / "chroma_db")
CHUNK_SIZE = 500
OVERLAP = 100
MIN_SCORE = 0.3          # tune on your own data

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(
    "notes", metadata={"hnsw:space": "cosine"}   # distance = 1 - cosine similarity
)


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks, start = [], 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap
    return chunks


def build_index():
    ids, docs, metas = [], [], []
    files = sorted(list(DATA_DIR.glob("*.txt")) + list(DATA_DIR.glob("*.md")))
    for path in files:
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{path.name}-{i}")
            docs.append(chunk)
            metas.append({"source": path.name, "chunk": i})
    if not docs:
        print("No documents found in data/")
        return
    embeddings = model.encode(docs, normalize_embeddings=True).tolist()
    collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    print(f"Indexed {len(docs)} chunks from {len(files)} files")


def search(query, k=3, min_score=MIN_SCORE):
    q = model.encode(query, normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=[q], n_results=k)
    results = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        score = 1 - dist
        if score >= min_score:
            results.append((doc, meta, score))
    return results


def main():
    build_index()
    print("Ask about your notes (type 'quit' to exit)")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue
        results = search(query)
        if not results:
            print("No relevant chunks found.")
            continue
        for doc, meta, score in results:
            print(f"\n[{score:.2f}] {meta['source']} (chunk {meta['chunk']})")
            print(doc)


if __name__ == "__main__":
    main()