"""Verify extraction run files for errors and null counts."""
import json

for f in ["evals/runs/glm-5.2.json", "evals/runs/deepseek-v4-pro-0813.json", "evals/runs/qwen3.5-397b.json"]:
    data = json.load(open(f))
    for doc_id, ext in data.get("extractions", {}).items():
        if isinstance(ext, dict) and ext.get("_error"):
            print(f"{f}: {doc_id} has ERROR: {ext['_error']}")
        else:
            n_none = sum(1 for v in ext.values() if v is None)
            print(f"{f}: {doc_id} OK ({len(ext)} fields, {n_none} null)")