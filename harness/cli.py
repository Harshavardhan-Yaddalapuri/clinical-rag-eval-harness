"""CLI entry point for the Clinical RAG Eval Harness.

Usage:
  python -m harness.cli retrieval-eval [--doc all|<doc_id>] [--k 5]
  python -m harness.cli extract --model <id> [--doc all|<doc_id>]
  python -m harness.cli eval [--all-models] [--regression] [--mock-run]
"""

from __future__ import annotations

import argparse
import os
import sys

from .retrieval_eval import (
    K,
    get_golden_doc_ids as get_retrieval_doc_ids,
    run_retrieval_eval,
    write_eval_output,
    print_eval_table,
)
from .extract import (
    get_golden_doc_ids,
    load_schema,
    run_extraction,
    save_run,
    list_run_models,
)
from .eval import (
    evaluate_all_models,
    build_summary,
    write_results,
    write_summary,
    load_baseline,
    check_regression,
    DEFAULT_REGRESSION_TOLERANCE,
)

EVAL_OUTPUT_PATH = os.path.join("evals", "retrieval.json")


def cmd_retrieval_eval(args: argparse.Namespace) -> int:
    """Run retrieval evaluation and write results."""
    if args.doc == "all" or args.doc is None:
        doc_ids = get_retrieval_doc_ids()
    else:
        doc_ids = [args.doc]

    if not doc_ids:
        print("ERROR: no golden documents found in data/golden/", file=sys.stderr)
        return 1

    k = args.k if args.k else K
    print(f"Running retrieval eval: docs={doc_ids}, k={k}")
    print()

    output = run_retrieval_eval(doc_ids=doc_ids, k=k)
    path = write_eval_output(output, EVAL_OUTPUT_PATH)
    print_eval_table(output)

    print()
    print(f"Written to {path}")

    for strat in output["strategies"]:
        m = strat["metrics"]
        print(
            f"  {strat['name']:<10}  hit@{k}={m['hit_at_k']:.4f}  "
            f"recall@{k}={m['recall_at_k']:.4f}  MRR={m['mrr']:.4f}  "
            f"n={m.get('n_queries', 0)}"
        )

    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    """Run LLM extraction for one or all docs and write a run file."""
    if args.doc == "all" or args.doc is None:
        doc_ids = get_golden_doc_ids()
    else:
        doc_ids = [args.doc]

    if not doc_ids:
        print("ERROR: no golden documents found in data/golden/", file=sys.stderr)
        return 1

    print(f"Extracting with model={args.model}: docs={doc_ids}")
    run = run_extraction(model=args.model, doc_ids=doc_ids)
    path = save_run(run)
    print(f"Written to {path}")
    for doc_id in doc_ids:
        n_fields = len(run["extractions"].get(doc_id, {}))
        print(f"  {doc_id}: {n_fields} fields extracted")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Score committed runs. --mock-run replays runs with no judge (no API)."""
    schema = load_schema()

    if args.mock_run:
        # Replay committed runs; judge is None so free_text falls back to
        # normalized exact match. No network, no API key.
        models = list_run_models()
        if not models:
            print("ERROR: no committed runs in evals/runs/", file=sys.stderr)
            return 1
        results = evaluate_all_models(schema, judge=None, models=models)
    elif args.all_models:
        results = evaluate_all_models(schema, judge=None)
    else:
        # Default: score every committed run (same as --all-models).
        results = evaluate_all_models(schema, judge=None)

    if not results:
        print("ERROR: no committed runs to evaluate", file=sys.stderr)
        return 1

    summary = build_summary(results)
    write_results(results)
    write_summary(summary)

    print("Extraction eval results:")
    for model, agg in summary.items():
        print(
            f"  {model:<28}  p={agg['precision']:.4f}  "
            f"r={agg['recall']:.4f}  f1={agg['f1']:.4f}  "
            f"docs={agg.get('n_documents', 0)}"
        )

    if args.regression:
        baseline = load_baseline()
        if not baseline:
            print("WARNING: no baseline at evals/baseline.json; skipping regression")
            return 0
        report = check_regression(summary, baseline, DEFAULT_REGRESSION_TOLERANCE)
        failed = False
        print()
        print("Regression check (tolerance 0.03):")
        for model, r in report.items():
            status = "PASS" if r["pass"] else "FAIL"
            if not r["pass"]:
                failed = True
            note = r.get("note", "")
            print(
                f"  {model:<28}  {status}  "
                f"f1={r.get('current_f1', 0):.4f} vs "
                f"baseline={r.get('baseline_f1', 0):.4f} "
                f"delta={r.get('delta', 0):+.4f} {note}"
            )
        if failed:
            print("Regression FAILED: one or more models below baseline")
            return 1

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness.cli",
        description="Clinical RAG Eval Harness CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # retrieval-eval
    re_parser = sub.add_parser(
        "retrieval-eval",
        help="Score retrieval strategies against golden queries",
    )
    re_parser.add_argument(
        "--doc",
        default="all",
        help="Document ID to eval (default: all golden docs)",
    )
    re_parser.add_argument(
        "--k",
        type=int,
        default=K,
        help=f"k for hit@k, recall@k, MRR (default: {K})",
    )
    re_parser.set_defaults(func=cmd_retrieval_eval)

    # extract
    ext_parser = sub.add_parser(
        "extract",
        help="Run extraction model against golden docs",
    )
    ext_parser.add_argument("--model", required=True, help="Model ID")
    ext_parser.add_argument("--doc", default="all", help="Document ID or 'all'")
    ext_parser.set_defaults(func=cmd_extract)

    # eval
    eval_parser = sub.add_parser(
        "eval",
        help="Score extraction results",
    )
    eval_parser.add_argument("--all-models", action="store_true")
    eval_parser.add_argument("--regression", action="store_true")
    eval_parser.add_argument("--mock-run", action="store_true")
    eval_parser.set_defaults(func=cmd_eval)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if hasattr(args, "func"):
        return args.func(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
