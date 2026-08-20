"""CLI entry point for the Clinical RAG Eval Harness.

Usage:
  python -m harness.cli retrieval-eval [--doc all|<doc_id>] [--k 5]
  python -m harness.cli extract --model <id> [--doc <doc_id>]
  python -m harness.cli eval [--all-models] [--regression] [--mock-run]

Currently implements: retrieval-eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .retrieval_eval import (
    K,
    get_golden_doc_ids,
    run_retrieval_eval,
    write_eval_output,
    print_eval_table,
)

EVAL_OUTPUT_PATH = os.path.join("evals", "retrieval.json")


def cmd_retrieval_eval(args: argparse.Namespace) -> int:
    """Run retrieval evaluation and write results."""
    if args.doc == "all" or args.doc is None:
        doc_ids = get_golden_doc_ids()
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

    # Honest summary
    for strat in output["strategies"]:
        m = strat["metrics"]
        print(
            f"  {strat['name']:<10}  hit@{k}={m['hit_at_k']:.4f}  "
            f"recall@{k}={m['recall_at_k']:.4f}  MRR={m['mrr']:.4f}  "
            f"n={m.get('n_queries', 0)}"
        )

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

    # extract (stub for future T4)
    ext_parser = sub.add_parser(
        "extract",
        help="Run extraction model against golden docs (T4, not yet implemented)",
    )
    ext_parser.add_argument("--model", required=True, help="Model ID")
    ext_parser.add_argument("--doc", default="all", help="Document ID or 'all'")

    # eval (stub for future T5)
    eval_parser = sub.add_parser(
        "eval",
        help="Score extraction results (T5, not yet implemented)",
    )
    eval_parser.add_argument("--all-models", action="store_true")
    eval_parser.add_argument("--regression", action="store_true")
    eval_parser.add_argument("--mock-run", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if hasattr(args, "func"):
        return args.func(args)

    # Stubs for not-yet-implemented commands
    if args.command == "extract":
        print("extract: not yet implemented (T4). Use retrieval-eval for now.")
        return 1
    if args.command == "eval":
        print("eval: not yet implemented (T5). Use retrieval-eval for now.")
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())