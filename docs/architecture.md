# Architecture — Scaling to Millions of Documents

**Date:** 2026-08-20 · **Status:** Design (reference for production deployment)

This document is the production-scale answer to the question the demo raises: how does
an eval harness (and the pipeline it gates) hold up when the corpus is millions of
documents and extraction must keep working across studies, sites, and model swaps?
The demo harness proves the *method*; this document is the *scale design*.

## 1. The core insight: clinical research is naturally sharded

Trials are isolated by law and by practice: a protocol, its amendments, its eTMF, its
site documents, and its source data belong to one study. Sponsors, CROs, and sites all
think in study boundaries. So the natural unit of scale is the **study shard** —
not "the corpus."

- **Isolation:** study A never queries study B's chunks (regulatory + access control
  boundary for free).
- **Scale:** adding a trial = adding a shard. 1,000 trials = 1,000 independent indexes.
- **Eval:** each shard gets its own golden set and its own regression gate — a new
  protocol cannot silently regress another trial's extraction.

## 2. Logical components at scale

```
┌─────────────────────────────────────────────────────────────────┐
│ Ingestion workers (async, per-study queue)                       │
│  pull protocol/amendments/eTMF docs → parse (OCR/layout) →      │
│  structural chunking (headers, tables, figures) → metadata →    │
│  embed (local/serving) → upsert into study shard                │
└──────────────┬──────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Index layer (per-study shard)                                    │
│  • Dense: HNSW ANN index (pgvector / Qdrant / Vespa)             │
│  • Sparse: BM25 inverted index                                   │
│  • Hybrid: RRF fusion at query time (k≈60)                       │
│  • Metadata filters: doc_type, section, visit, arm, amendment    │
└──────────────┬──────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Query service                                                    │
│  query → optional query rewrite/decomposition → per-shard        │
│  retrieval → rerank (optional) → context assembly → LLM answer   │
└──────────────┬──────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Eval & governance (THE LAYER THIS HARNESS DEMOS)                 │
│  • golden set per study (extraction fields + retrieval queries)  │
│  • nightly/CI eval: all models × all studies → score tables      │
│  • regression gate: model/prompt/chunking change must not drop   │
│    below baseline (this repo's gate, per-shard)                  │
│  • audit: every run stored (model, prompt, inputs, outputs,      │
│    scores, reviewer edits) for 21 CFR Part 11-style traceability │
│  • drift: sampling + reviewer feedback feeds retraining/RL        │
└──────────────────────────────────────────────────────────────────┘
```

## 3. Ingestion pipeline

- **Async by default** — ingestion is I/O-bound; synchronous ingestion is an anti-pattern
  (RAG-Engineer rule). Per-study queue; workers scale horizontally; idempotent upserts
  (doc_hash content addressing) so re-runs never duplicate.
- **Chunk for retrieval, not ingestion** — structural chunking for protocols (headers,
  tables, Schedule of Activities), with section-path metadata. The chunk unit is the
  retrieval unit, not the storage unit.
- **Amendments as first-class** — a protocol amendment creates a new versioned snapshot;
  chunks carry version + effective date; retrieval prefers current version unless asked
  otherwise; amendment deltas are indexable ("what changed" queries).

## 4. Index layer (per-study shard)

- **Dense:** HNSW index over chunk embeddings. `ef_construction`/`m` tuned per corpus
  (latency/recall tradeoff). Embedding model validated against the *actual* corpus —
  MTEB winners can underperform on clinical jargon (RAG-Engineer rule: validate
  embeddings on your corpus).
- **Sparse:** BM25 as the lexical baseline; clinical terms (drug names, lab codes,
  ICD/MedDRA codes) benefit from exact token match.
- **Hybrid:** RRF fusion (k≈60) — consistent with the best-practices survey
  (arXiv:2407.01219) and what this repo measures.
- **Metadata filters before semantic search:** study, section, arm, visit, amendment —
  scope first, then search (RAG-Engineer rule).

## 5. Storage & serving

| Option | When | Notes |
|---|---|---|
| Postgres + pgvector (HNSW) | default | single system for chunks, eval results, audit |
| Qdrant / Milvus | dedicated vector serving | higher QPS, disk ANN, distributed |
| Object store + tiered ANNS | archival/cold shards | distributed ANNS on object storage (arXiv:2510.17326) |

"Less LLM, more documents" (arXiv:2510.02657) is the strategic lever: retrieval quality
and corpus scale move the needle more than generator size. The eval harness is the
instrument that keeps retrieval quality honest as the corpus grows.

## 6. The eval gate at scale (this repo, in production shape)

- Golden sets per study shard (this repo: 3 studies × 5 synthesis queries).
- CI per change: pytest → mock-run (replayed runs) → regression gate (≤0.03 drop from
  baseline) → deploy. No API keys in CI.
- Nightly eval of all studies on new model versions — a model that wins on the demo
  study may regress on oncology; the per-shard gate catches it.
- Reviewer corrections collected as preference data (RL line in the JD): human accepts
  /overrides in the review UI → logged → sampled → used for fine-tuning/RL.

## 7. Observability & drift

- Run logging: model, prompt hash, retrieval ids, scores, latency, cost.
- Drift detection: query-log distribution, retrieval hit-rate, reviewer correction
  rate by study.
- Alerts on: regression gate breach, ingestion lag, embedding/model version skew.

## 8. Security / regulatory posture

- Tenant isolation by study shard (access control list per study).
- No PHI in eval harness (public trials only in this repo); production pipelines run
  inside customer-controlled cloud (air-gapped option), consistent with the industry's
  zero-trust deployments (see InteriMed/coTrial positioning, market scan 2026-08).
- Audit trail for 21 CFR Part 11 alignment.

## 9. References
- SANE (arXiv:2608.00658) — retrieval vs reading failure taxonomy.
- Searching for Best Practices in RAG (arXiv:2407.01219).
- BEIR (arXiv:2104.08663) — zero-shot IR evaluation.
- RAPTOR (arXiv:2401.18059) — hierarchical summarization retrieval.
- Late Chunking (arXiv:2409.04701) — contextual chunk embeddings.
- Less LLM, More Documents (arXiv:2510.02657) — corpus scaling beats generator scaling.
- HAKES (arXiv:2505.12524) — load-balanced ANN index service.
- Distributed ANNS on object storage (arXiv:2510.17326).
- LoCo/M2-BERT (arXiv:2402.07440) — long-document retrieval benchmark.
