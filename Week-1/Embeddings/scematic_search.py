from sentence_transformers import SentenceTransformer, util

MY_SKILLS = ["python", "java", "sql", "power bi", "pyspark", "rag",
             "scikit-learn", "prophet", "git", "json", "rest api"]

model = SentenceTransformer("all-MiniLM-L6-v2")
my_emb = model.encode(MY_SKILLS)

THRESHOLD = 0.5   # you will tune this


def best_match(required_skill):
    emb = model.encode(required_skill)
    scores = util.cos_sim(emb, my_emb)[0]
    idx = int(scores.argmax())
    return MY_SKILLS[idx], float(scores[idx])


required = ["python basics", "javascript", "calling web services",
            "data visualization", "sql server", "machine learning frameworks"]

for skill in required:
    mine, score = best_match(skill)
    status = "HAVE" if score >= THRESHOLD else "MISSING"
    print(f"{skill:30} -> {mine:14} {score:.2f}  {status}")