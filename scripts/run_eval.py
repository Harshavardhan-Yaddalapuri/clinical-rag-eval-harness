"""Run full extraction eval with LLM judge for free_text fields.

Scores all committed runs (3 models x 3 docs) using rule-based scorers
for structured fields and gpt-oss:20b as the LLM judge for any free_text
fields. Writes:
  - evals/results.json   (per-model, per-doc, per-field detail)
  - evals/summary.json   (per-model aggregates: precision, recall, f1)
  - evals/baseline.json  (first-run snapshot for regression detection)

Usage:
  source ~/.hermes/.env && python scripts/run_eval.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.llm_client import LLMClient
from harness.extract import load_schema, list_run_models
from harness.eval import (
    evaluate_model,
    build_summary,
    write_results,
    write_summary,
    load_baseline,
    check_regression,
    BASELINE_PATH,
    DEFAULT_REGRESSION_TOLERANCE,
)

JUDGE_MODEL = "gpt-oss:20b"
SLEEP_BETWEEN_MODELS = 3.0


def main() -> int:
    if not os.environ.get("OLLAMA_API_KEY"):
        print("ERROR: OLLAMA_API_KEY not set in environment", file=sys.stderr)
        return 1

    schema = load_schema()
    models = list_run_models()
    if not models:
        print("ERROR: no committed runs in evals/runs/", file=sys.stderr)
        return 1

    print(f"Models to evaluate: {models}")
    print(f"Judge model: {JUDGE_MODEL}")

    # Create judge client (used for free_text fields if any exist)
    judge = LLMClient.from_env(JUDGE_MODEL, timeout_s=90, max_retries=3)

    results = {}
    for model in models:
        print(f"\nEvaluating {model}...", end=" ", flush=True)
        try:
            result = evaluate_model(model, schema, judge=judge)
            results[model] = result
            agg = result["aggregates"]
            print(f"p={agg['precision']:.4f} r={agg['recall']:.4f} f1={agg['f1']:.4f} "
                  f"(docs={agg.get('n_documents', 0)})")
        except Exception as exc:
            print(f"ERROR: {exc}")
            results[model] = {"model": model, "error": str(exc), "aggregates": {"precision": 0, "recall": 0, "f1": 0, "n_documents": 0}}
        time.sleep(SLEEP_BETWEEN_MODELS)

    # Write results and summary
    write_results(results)
    summary = build_summary(results)
    write_summary(summary)

    # Write baseline (snapshot of first full run)
    # Only update baseline if it doesn't already cover all current models
    existing_baseline = load_baseline()
    needs_update = False
    for model in summary:
        if model not in existing_baseline:
            needs_update = True
            break
    if needs_update or len(existing_baseline) < len(summary):
        with open(BASELINE_PATH, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nBaseline written to {BASELINE_PATH}")

    # Print summary table
    print("\n" + "=" * 70)
    print("EXTRACTION EVAL SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Docs':>5}")
    print("-" * 70)
    for model, agg in sorted(summary.items()):
        print(f"{model:<30} {agg['precision']:>10.4f} {agg['recall']:>10.4f} "
              f"{agg['f1']:>10.4f} {agg.get('n_documents', 0):>5}")

    # Per-doc breakdown
    print("\nPer-document F1:")
    for model, result in sorted(results.items()):
        if "per_document" not in result:
            continue
        for doc_id, doc_result in sorted(result["per_document"].items()):
            doc_agg = doc_result["aggregates"]
            print(f"  {model:<28} {doc_id:<8} f1={doc_agg['f1']:.4f} "
                  f"(p={doc_agg['precision']:.4f} r={doc_agg['recall']:.4f})")

    # Regression check
    if existing_baseline:
        report = check_regression(summary, existing_baseline, DEFAULT_REGRESSION_TOLERANCE)
        print("\nRegression check (tolerance 0.03):")
        any_fail = False
        for model, r in sorted(report.items()):
            status = "PASS" if r["pass"] else "FAIL"
            if not r["pass"]:
                any_fail = True
            note = r.get("note", "")
            delta = r.get("delta", 0)
            print(f"  {model:<30} {status}  delta={delta:+.4f} {note}")
        if any_fail:
            print("\nWARNING: regression detected on one or more models")

    print("\nResults: evals/results.json")
    print("Summary: evals/summary.json")
    print("Baseline: evals/baseline.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())