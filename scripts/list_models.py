"""List available models from the Ollama cloud registry."""
import os
import sys

import requests

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1")
API_KEY = os.environ.get("OLLAMA_API_KEY")

headers = {"Content-Type": "application/json"}
if API_KEY:
    headers["Authorization"] = f"Bearer {API_KEY}"

resp = requests.get(f"{BASE_URL}/models", headers=headers, timeout=30)
print(f"Status: {resp.status_code}", file=sys.stderr)
data = resp.json()
models = []
if isinstance(data, dict) and "data" in data:
    models = [m.get("id", "") for m in data["data"]]
elif isinstance(data, list):
    models = [m.get("id", "") if isinstance(m, dict) else str(m) for m in data]

for m in sorted(models):
    print(m)