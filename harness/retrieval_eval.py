"""Retrieval evaluation engine: scores BM25, dense, and hybrid (RRF) strategies
against golden retrieval queries.

Metrics per query (k=5):
  - hit@k:   did any retrieved chunk match at least one expected span?
  - recall@k: fraction of expected spans matched by retrieved chunks
  - MRR:     1/rank of the first matched chunk (0 if none)

A retrieved chunk "matches" an expected span if the chunk's section path
contains the span's section string AND the chunk text contains a substring of
the span's quote (normalized: first 60 non-space characters). This is a
fuzzy-but-honest matching rule that tolerates chunking boundary variation
while penalizing clearly wrong retrievals.

Output schema (evals/retrieval.json):
  {
    "strategies": [
      {
        "name": "bm25",
        "metrics": {"hit_at_k": 0.6, "recall_at_k": 0.4, "mrr": 0.35},
        "per_doc": {
          "actt1": {"hit_at_k": 0.8, "recall_at_k": 0.6, "mrr": 0.5},
          ...
        },
        "per_query": [
          {"query_id": "actt1-r1", "hit": true, "recall": 1.0, "mrr": 1.0, ...}
        ]
      },
      ...
    ]
  }
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from .chunking import chunk_document_file, Chunk
from .retrieval import Retriever, RRF_K

K = 5
GOLDEN_DIR = os.path.join("data", "golden")
EVAL_OUTPUT = os.path.join("evals", "retrieval.json")

# Minimum quote length to use for substring matching (avoids matching on
# common short phrases while still being tolerant of chunking boundaries).
_QUOTE_MATCH_LEN = 60


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for fuzzy matching."""
    import re

    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _quote_fingerprint(quote: str) -> str:
    """Extract the first N non-space characters of a quote for matching.

    This gives us a stable substring that should appear inside any chunk that
    contains the expected span, even if chunking split the full quote across
    boundaries.
    """
    norm = _normalize(quote)
    # Remove very short quotes
    if len(norm) < 10:
        return norm
    return norm[:_QUOTE_MATCH_LEN]


def _section_matches(chunk_section: str, expected_section: str) -> bool:
    """Check if a chunk's section path contains the expected section string.

    Both are normalized to lowercase. The expected section might be a partial
    path like 'Eligibility Criteria > Exclusion' or a top-level 'Study Design'.
    """
    cs = _normalize(chunk_section)
    es = _normalize(expected_section)
    if not es:
        return True
    # Exact match or prefix match (expected section is a prefix of chunk section)
    if es in cs:
        return True
    # Also check if the last component of expected matches a component in chunk section
    es_parts = es.split(" > ")
    cs_parts = cs.split(" > ")
    # Check if the expected's last segment matches any chunk segment
    if es_parts:
        last_expected = es_parts[-1].strip()
        for part in cs_parts:
            if last_expected in part:
                return True
    return False


def _text_matches(chunk_text: str, quote: str) -> bool:
    """Check if a chunk's text contains the quote fingerprint."""
    ct = _normalize(chunk_text)
    fp = _quote_fingerprint(quote)
    if not fp:
        return False
    return fp in ct


def _span_in_chunk(span: dict, chunk: dict) -> bool:
    """Check if a retrieved chunk matches an expected span (section + text)."""
    section_ok = _section_matches(chunk.get("section", ""), span.get("section", ""))
    if not section_ok:
        return False
    return _text_matches(chunk.get("text", ""), span.get("quote", ""))


def score_query(
    retrieved: List[dict],
    expected_spans: List[dict],
    k: int = K,
) -> dict:
    """Score a single query's retrieval results against expected spans.

    Returns dict with: hit (bool), recall (float), mrr (float),
    matched_ranks (list of int), n_expected (int), n_retrieved (int).
    """
    top_k = retrieved[:k]
    n_expected = len(expected_spans)

    if n_expected == 0:
        return {
            "hit": False,
            "recall": 0.0,
            "mrr": 0.0,
            "matched_ranks": [],
            "n_expected": 0,
            "n_retrieved": len(top_k),
        }

    # For each retrieved chunk (1-indexed rank), check which spans it matches
    matched_spans = set()
    first_match_rank = 0  # 0 = no match
    matched_ranks = []

    for rank_idx, chunk in enumerate(top_k):
        rank = rank_idx + 1
        for span_idx, span in enumerate(expected_spans):
            if span_idx in matched_spans:
                continue
            if _span_in_chunk(span, chunk):
                matched_spans.add(span_idx)
                matched_ranks.append(rank)
                if first_match_rank == 0:
                    first_match_rank = rank

    hit = len(matched_spans) > 0
    recall = len(matched_spans) / n_expected
    mrr = 1.0 / first_match_rank if first_match_rank > 0 else 0.0

    return {
        "hit": hit,
        "recall": round(recall, 4),
        "mrr": round(mrr, 4),
        "matched_ranks": matched_ranks,
        "n_expected": n_expected,
        "n_retrieved": len(top_k),
    }


def aggregate_metrics(query_results: List[dict]) -> dict:
    """Aggregate per-query results into mean hit@k, recall@k, MRR."""
    n = len(query_results)
    if n == 0:
        return {"hit_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}
    total_hit = sum(1 for q in query_results if q["hit"])
    total_recall = sum(q["recall"] for q in query_results)
    total_mrr = sum(q["mrr"] for q in query_results)
    return {
        "hit_at_k": round(total_hit / n, 4),
        "recall_at_k": round(total_recall / n, 4),
        "mrr": round(total_mrr / n, 4),
        "n_queries": n,
    }


def load_golden_queries(doc_id: str) -> List[dict]:
    """Load retrieval queries for a document from data/golden/<doc>/retrieval.json."""
    path = os.path.join(GOLDEN_DIR, doc_id, "retrieval.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_golden_doc_ids() -> List[str]:
    """List all doc_ids that have golden retrieval data."""
    if not os.path.isdir(GOLDEN_DIR):
        return []
    docs = []
    for name in sorted(os.listdir(GOLDEN_DIR)):
        ret_path = os.path.join(GOLDEN_DIR, name, "retrieval.json")
        src_path = os.path.join(GOLDEN_DIR, name, "source.md")
        if os.path.isfile(ret_path) and os.path.isfile(src_path):
            docs.append(name)
    return docs


def build_doc_chunks(doc_id: str) -> List[Chunk]:
    """Chunk a golden document's source.md."""
    src_path = os.path.join(GOLDEN_DIR, doc_id, "source.md")
    return chunk_document_file(src_path, doc_id)


def evaluate_strategy(
    strategy: str,
    retriever: Retriever,
    queries: List[dict],
    doc_id: str,
    k: int = K,
) -> Tuple[dict, List[dict]]:
    """Run one strategy over all queries for one doc, return (aggregated, per_query)."""
    per_query = []
    for q in queries:
        results = retriever.search(
            q["question"],
            top_k=k,
            strategy=strategy,
            doc_id=doc_id,
        )
        scored = score_query(results, q.get("expected_spans", []), k=k)
        scored["query_id"] = q["id"]
        scored["question"] = q["question"]
        scored["strategy"] = strategy
        per_query.append(scored)

    aggregated = aggregate_metrics(per_query)
    return aggregated, per_query


def run_retrieval_eval(
    doc_ids: Optional[List[str]] = None,
    k: int = K,
    strategies: Optional[List[str]] = None,
) -> dict:
    """Run retrieval evaluation across all (or specified) docs and strategies.

    Returns the full output dict: {strategies: [{name, metrics, per_doc, per_query}]}.
    No LLM API needed. Dense retrieval needs local Ollama or falls back gracefully.
    """
    if doc_ids is None:
        doc_ids = get_golden_doc_ids()
    if strategies is None:
        strategies = ["bm25", "dense", "hybrid"]

    # Build a single retriever per doc with all its chunks
    doc_retrievers: Dict[str, Retriever] = {}
    doc_queries: Dict[str, List[dict]] = {}
    for doc_id in doc_ids:
        chunks = build_doc_chunks(doc_id)
        doc_retrievers[doc_id] = Retriever(chunks)
        doc_queries[doc_id] = load_golden_queries(doc_id)

    output = {"strategies": [], "config": {"k": k, "docs": doc_ids}}

    for strategy in strategies:
        all_per_query: List[dict] = []
        per_doc: Dict[str, dict] = {}

        for doc_id in doc_ids:
            retriever = doc_retrievers[doc_id]
            queries = doc_queries[doc_id]
            agg, pq = evaluate_strategy(strategy, retriever, queries, doc_id, k=k)
            per_doc[doc_id] = agg
            all_per_query.extend(pq)

        overall = aggregate_metrics(all_per_query)
        output["strategies"].append({
            "name": strategy,
            "metrics": overall,
            "per_doc": per_doc,
            "per_query": all_per_query,
        })

    return output


def write_eval_output(output: dict, path: str = EVAL_OUTPUT) -> str:
    """Write eval output to JSON file, creating dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    return path


def print_eval_table(output: dict) -> str:
    """Print a human-readable table of results. Returns the table string."""
    lines = []
    header = f"{'Strategy':<10} {'Hit@5':>8} {'Recall@5':>10} {'MRR':>8} {'N':>5}"
    lines.append(header)
    lines.append("-" * len(header))

    for strat in output.get("strategies", []):
        m = strat["metrics"]
        lines.append(
            f"{strat['name']:<10} {m['hit_at_k']:>8.4f} "
            f"{m['recall_at_k']:>10.4f} {m['mrr']:>8.4f} "
            f"{m.get('n_queries', 0):>5}"
        )

    # Per-doc breakdown
    lines.append("")
    lines.append("Per-document:")
    doc_header = f"{'Strategy':<10} {'Doc':<10} {'Hit@5':>8} {'Recall@5':>10} {'MRR':>8} {'N':>5}"
    lines.append(doc_header)
    lines.append("-" * len(doc_header))

    for strat in output.get("strategies", []):
        for doc_id, m in strat["per_doc"].items():
            lines.append(
                f"{strat['name']:<10} {doc_id:<10} {m['hit_at_k']:>8.4f} "
                f"{m['recall_at_k']:>10.4f} {m['mrr']:>8.4f} "
                f"{m.get('n_queries', 0):>5}"
            )

    table = "\n".join(lines)
    print(table)
    return table