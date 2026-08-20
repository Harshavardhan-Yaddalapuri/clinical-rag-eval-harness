"""Run real model extractions for all 3 models across all 3 golden docs.

Extracts fields using glm-5.2, deepseek-v4-pro:0813, qwen3.5:397b.
Handles per-model failures gracefully (records error JSON, continues).
Sleeps between calls for rate limiting. Caches per (model, doc) so
re-runs only fetch missing docs.

Usage:
  source ~/.hermes/.env && python scripts/run_extractions.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.llm_client import LLMClient, LLMError, LLMJSONError, LLMRateLimit
from harness.extract import (
    load_schema,
    get_golden_doc_ids,
    extract_fields,
    load_run,
    save_run,
    GOLDEN_DIR,
    PROMPT_PATH,
)

MODELS = ["glm-5.2", "deepseek-v4-pro:0813", "qwen3.5:397b"]
SLEEP_BETWEEN_DOCS = 2.0  # seconds between docs (rate limit)
SLEEP_BETWEEN_MODELS = 5.0  # seconds between models
EXTRACTION_TIMEOUT = 120  # per-request timeout


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_one_model(model: str, doc_ids: list, schema: dict) -> dict:
    """Extract all docs for one model. Gracefully handles per-doc errors."""
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"Docs: {doc_ids}")
    print(f"{'='*60}")

    client = LLMClient.from_env(model, timeout_s=EXTRACTION_TIMEOUT, max_retries=3)
    run = load_run(model)  # merge into existing

    for doc_id in doc_ids:
        # Skip if already extracted (cache hit)
        if doc_id in run.get("extractions", {}):
            existing = run["extractions"][doc_id]
            if isinstance(existing, dict) and existing.get("_error"):
                print(f"  [{doc_id}] cached ERROR, re-trying...")
            else:
                print(f"  [{doc_id}] cached, skipping")
                continue

        src_path = os.path.join(GOLDEN_DIR, doc_id, "source.md")
        with open(src_path, "r", encoding="utf-8") as fh:
            source_text = fh.read()

        print(f"  [{doc_id}] extracting ({len(source_text)} chars)...", end=" ", flush=True)
        try:
            result = extract_fields(doc_id, source_text, schema, client, PROMPT_PATH)
            run["extractions"][doc_id] = result
            n_fields = sum(1 for v in result.values() if v is not None)
            print(f"OK ({n_fields} non-null fields)")
        except (LLMError, LLMJSONError, LLMRateLimit) as exc:
            run["extractions"][doc_id] = {"_error": str(exc), "_error_type": type(exc).__name__}
            print(f"ERROR: {exc}")
        except Exception as exc:
            run["extractions"][doc_id] = {"_error": str(exc), "_error_type": type(exc).__name__}
            print(f"UNEXPECTED ERROR: {exc}")
            traceback.print_exc()

        # Save after each doc so progress survives crashes
        run["model"] = model
        run["created_at"] = now_iso()
        save_run(run)

        time.sleep(SLEEP_BETWEEN_DOCS)

    return run


def main() -> int:
    # Verify API key is set
    if not os.environ.get("OLLAMA_API_KEY"):
        print("ERROR: OLLAMA_API_KEY not set in environment", file=sys.stderr)
        return 1

    schema = load_schema()
    doc_ids = get_golden_doc_ids()
    print(f"Golden docs: {doc_ids}")
    print(f"Models: {MODELS}")

    for model in MODELS:
        run = run_one_model(model, doc_ids, schema)
        n_ok = sum(1 for v in run.get("extractions", {}).values()
                   if isinstance(v, dict) and not v.get("_error"))
        n_err = sum(1 for v in run.get("extractions", {}).values()
                    if isinstance(v, dict) and v.get("_error"))
        print(f"\n  {model}: {n_ok} OK, {n_err} errors out of {len(doc_ids)} docs")
        time.sleep(SLEEP_BETWEEN_MODELS)

    print("\n\nAll extractions complete.")
    print("Run files in evals/runs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())