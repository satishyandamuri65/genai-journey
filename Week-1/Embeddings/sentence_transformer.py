from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = ["REST API: a way for programs to communicate over HTTP",
         "calling web services to fetch data from a server",]

embeddings = model.encode(texts)
print("Shape:", embeddings.shape)
scores = util.cos_sim(embeddings, embeddings)


for i, a in enumerate(texts):
    for j, b in enumerate(texts):
        if i < j:
            print(f"{a!r} vs {b!r}: {scores[i][j].item():.2f}")
            