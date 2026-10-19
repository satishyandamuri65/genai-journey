from llm import ask_llm
import re

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "role_title": {"type": "STRING"},
        "required_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
        "nice_to_have": {"type": "ARRAY", "items": {"type": "STRING"}},
        "freshers_welcome": {"type": "BOOLEAN"},
    },
}

MY_SKILLS = {"python", "java", "sql", "power bi", "pyspark", "rag",
             "scikit-learn", "prophet", "git", "json", "rest api"}

ALIASES = {"rest apis": "rest api", "powerbi": "power bi", "sklearn": "scikit-learn"}


def normalize(skill):
    s = skill.lower().strip()
    return ALIASES.get(s, s)


JOBS = {
    "Clarion intern": """Intern GenAI. Python, REST APIs, JSON, RAG, embeddings,
    prompt engineering. Good to have: PyTorch, Hugging Face, OpenCV, Docker.""",
    "Sample analyst": """Data Analyst fresher. SQL, Power BI, Excel, JavaScript, Python basics.
    Nice to have: Tableau.""",
}

def has_skill(required_skill):
    skill_text = required_skill.lower()
    for mine in MY_SKILLS:
        if re.search(rf"\b{re.escape(mine)}\b", skill_text):
            return True
    return False


for name, text in JOBS.items():
    result = ask_llm(
        text,
        system="Extract information from job descriptions. Return only JSON.",
        temperature=0,
        schema=SCHEMA,
    )
    required = {normalize(s) for s in result["required_skills"]}
    have = {s for s in required if has_skill(s)}
    missing = required - have
    print(f"\n{name} ({result['role_title']})")
    print("  You have:", sorted(have))
    print("  Missing :", sorted(missing))
    print(f"  Match   : {len(have)}/{len(required)}")