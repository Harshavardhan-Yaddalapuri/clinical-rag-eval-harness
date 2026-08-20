"""Inspect eval JSON structures for the web viewer build."""
import json

with open("evals/retrieval.json") as f:
    d = json.load(f)

print("=== retrieval.json ===")
print("Top keys:", list(d.keys()))
print("Strategies:", [s["name"] for s in d["strategies"]])
pq = d["strategies"][0]["per_query"][0]
print("Per-query fields:", {k: type(v).__name__ for k, v in pq.items()})
print("Sample per_query:", json.dumps(pq, indent=2)[:800])
print()

with open("evals/results.json") as f:
    r = json.load(f)

model_key = list(r.keys())[0]
print("=== results.json ===")
print("Top-level keys (models):", list(r.keys()))
print("Model entry keys:", list(r[model_key].keys()))
doc_key = list(r[model_key]["per_document"].keys())[0]
print("Per-doc keys:", list(r[model_key]["per_document"][doc_key].keys()))
pf = r[model_key]["per_document"][doc_key]["per_field"]
print("Per-field keys:", list(pf.keys())[:10], "...")
sample_field = list(pf.keys())[1]
print("Sample field result:", json.dumps(pf[sample_field], indent=2))
print()

with open("evals/summary.json") as f:
    s = json.load(f)
print("=== summary.json ===")
print(json.dumps(s, indent=2))