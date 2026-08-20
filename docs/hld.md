# Clinical RAG Eval Harness — High-Level Design (HLD)

**Version:** 1.0.0 · **Date:** 2026-08-20 · **Status:** Implemented (iterative hardening)

## 1. Purpose

A production-grade **evaluation and regression harness** for LLM-based clinical document
pipelines — measuring *retrieval* quality (BM25 vs dense vs hybrid) and *extraction*
quality (structured field scoring across models), with a CI regression gate so pipeline
changes cannot silently degrade results at live trial sites.

The product answers the two RAG failure modes defined in SANE (arXiv:2608.00658):
retrieval failure (right span never recalled) and reading failure (right span, wrong
answer). Extraction eval alone catches half of it; this harness catches both.

## 2. Goals & Non-Goals

### Goals
- Golden-set driven evaluation for retrieval and extraction on real clinical documents.
- Model-agnostic extraction engine (registry-exact model IDs, pluggable via CLI).
- Deterministic scoring rules + LLM-as-judge for free-text fields.
- CI regression gate (replayed runs, zero API keys in CI).
- Mobile-first, zero-login viewer rendering committed eval artifacts.
- Honest, reproducible numbers: all artifacts committed; nothing fabricated.

### Non-Goals (deliberately out of scope)
- A RAG *runtime* (chunking/retrieval exists only to power the eval).
- Deploying clinical software; no PHI/PHI-adjacent data (public trials only).
- A scale benchmark: the golden set proves the method; the architecture doc
  (docs/architecture.md) carries the scale design.
- Any external LLM calls inside CI.

## 3. System Context

```
            ┌─────────────────────────────────────────────────────┐
            │                  Harsha / Reviewer                   │
            │        (LinkedIn InMail → live viewer link)          │
            └───────────────┬──────────────────┬──────────────────┘
                            │                  │
              (mobile, zero-login)        (live run — POST /api/run)
                            ▼                  ▼
                 ┌────────────────────────┐   ┌──────────────────┐
                 │   web/ (Next.js)       │   │  API route       │
                 │  scorecard-first view  │──▶│  (server-side    │
                 │  /documents/:id        │   │   OLLAMA key)    │
                 │  retrieval debug       │   └────────┬─────────┘
                 └───────────┬────────────┘            │
                             │ fs read (committed)     ▼
                             │               ┌──────────────────┐
                             │               │  https://ollama. │
                             │               │  com/v1 (cloud)   │
                             │               │  + local 11434   │
                             │               └──────────────────┘
                             ▼
        ┌────────────────────────────────────────────────────────┐
        │  harness/  (Python CLI)                                 │
        │  extract ─▶ eval ─▶ regression-gate                     │
        │  retrieval-eval (bm25 · dense · hybrid RRF)             │
        └──────────────┬─────────────────────────────────────────┘
                       │ commits artifacts
                       ▼
        ┌────────────────────────────────────────────────────────┐
        │  data/golden/   shared/schema.json   evals/ (committed)│
        └────────────────────────────────────────────────────────┘
                       ▲
        ┌───────────────┴────────────────────────────────────────┐
        │  GitHub Actions CI: pytest → mock-run → regression gate │
        │  (zero API keys)                                        │
        └─────────────────────────────────────────────────────────┘
```

## 4. Components

| Component | Location | Responsibility |
|---|---|---|
| Golden corpus | `data/golden/<doc>/` | `source.md` (protocol text), `golden.json` (hand-verified fields), `retrieval.json` (5 synthesis queries/doc with expected spans) |
| Shared contract | `shared/schema.json` | Single source of truth: doc metadata, field definitions + types, eval config, retrieval config, aggregation/gate config |
| Extraction engine | `harness/extract.py` | Chunk protocol → prompt (shared/prompts/extract.txt) → strict JSON per schema → null for unknown |
| Eval engine | `harness/eval.py` | Rule-based scoring (numeric tolerance, boolean exact, date fuzzy, categorical normalized) + LLM judge for free-text |
| Retrieval engine | `harness/retrieval.py` | BM25 (rank-bm25), dense (local embeddings), hybrid RRF; chunked, metadata-aware |
| Retrieval eval | `harness/retrieval_eval.py` | hit@k, recall@k, MRR per strategy/doc |
| CLI | `harness/cli.py` | extract / eval / retrieval-eval / regression / mock-run |
| Web viewer | `web/` | Next.js, scorecard-first, mobile-first, read-only JSON at request time |
| CI | `.github/workflows/ci.yml` | pytest + mock-run regression gate + node build |

## 5. Data Contracts

### shared/schema.json (canonical)
```json
{
  "version": "1.0.0",
  "documents": [{ "doc_id", "nct_id", "title", "disease_area", "fields": {key: {type, description, ...}} }],
  "eval_config": {
    "extraction": { "scoring": {numeric/boolean/date/categorical/free_text/list}, "judge_model", "models": [...] },
    "retrieval": { "strategies": ["bm25","dense","hybrid"], "metrics": ["hit@k","recall@k","mrr"], "k": 5, "embedding_backend", "embedding_fallback" }
  },
  "aggregation": { "metrics": ["precision","recall","f1"], "per_field": true, "per_document": true,
                   "baseline_file": "evals/baseline.json", "regression_tolerance": 0.03 }
}
```
Single source of truth: the web app renders whatever the harness produced; the harness
never hardcodes a schema the UI re-declares.

### evals/ artifacts (all committed — honesty by construction)
| File | Contents |
|---|---|
| `evals/runs/<model>.json` | raw candidate extraction per doc per model |
| `evals/results.json` | per-model × per-doc × per-field scores |
| `evals/summary.json` | model-level aggregate (precision/recall/f1) |
| `evals/baseline.json` | first-run scores; the gate reference |
| `evals/retrieval.json` | per-strategy × per-doc retrieval metrics |

## 6. Key Decisions (ADRs live in docs/Decisions.md)
1. **Golden set from real public trials** (ACTT-1, CheckMate-017, DAPA-CKD) — verifiable,
   no PHI, reproducible by any reviewer.
2. **Hybrid RRF over BM25 + dense** as the reference strategy (best-practices survey,
   arXiv:2407.01219).
3. **LLM-as-judge** for free-text fields with a strict, null-aware prompt; deterministic
   rules for everything else.
4. **Regression gate replays committed runs** — no API keys in CI, deterministic gating.
5. **Vercel, not LocalStack** — a live URL a founder can click, not a fake AWS.
6. **Shard-by-study scale design** (see docs/architecture.md) — the demo shows the method;
   the doc shows the scale.

## 7. Quality Bar
- Every module unit-tested (pytest, no network).
- Every CLI command verified by real execution (scores committed, nothing fabricated).
- `npm run build` green; viewer renders from committed artifacts.
- Architecture + decisions documented; all doc numbers traceable to source.
