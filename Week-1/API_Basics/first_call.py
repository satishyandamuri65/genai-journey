import os
import json
import importlib.util
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
body = {
    "contents": [{"role": "user", "parts": [{"text": "Explain what an API is in one sentence."}]}]
}

request = Request(
    url,
    data=json.dumps(body).encode("utf-8"),
    headers=headers,
    method="POST",
)

try: 
    with urlopen(request) as response:
        print(response.status)
        result = json.loads(response.read().decode("utf-8"))
        print(result["candidates"][0]["content"]["parts"][0]["text"])
except HTTPError as error:
    print(error.code)
    print(error.read().decode("utf-8"))