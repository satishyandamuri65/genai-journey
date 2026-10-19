import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"  # check the current name in Google's docs
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


def ask_llm(prompt, system=None, temperature=0.2, schema=None, retries=3):
    """Send a prompt to Gemini. Returns text, or a dict if a schema is given."""
    headers = {"x-goog-api-key": API_KEY, "Content-Type": "application/json"}
    config = {"temperature": temperature}
    if schema:
        config["responseMimeType"] = "application/json"
        config["responseSchema"] = schema

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": config,
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    last_error = ""
    for attempt in range(retries):
        r = requests.post(URL, headers=headers, json=body, timeout=60)
        if r.status_code == 429:
            last_error = r.text
            wait = 15 * (attempt + 1)          # 15s, 30s, 45s, 60s
            print(f"Rate limited (429). Waiting {wait}s...")
            time.sleep(wait)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"API error {r.status_code}: {r.text[:300]}")
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text) if schema else text

    raise RuntimeError(f"Still rate limited after retries. Last response: {last_error[:300]}")