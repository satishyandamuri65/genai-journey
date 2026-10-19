import json
from llm import ask_llm

JD = """We are hiring a Junior GenAI Engineer. You should know Python, SQL,
REST APIs and have basic knowledge of RAG and vector databases."""

valid = 0
for i in range(5):
    text = ask_llm(
        "Return the required skills from this job description as JSON.\n\n" + JD,
        temperature=1.0,
    )
    print(f"--- Run {i + 1} ---")
    print(text)
    try:
        json.loads(text)
        valid += 1
    except json.JSONDecodeError:
        print("json.loads failed")

print(f"\nValid JSON: {valid}/5")
