# LLD -- Modules & API Guide

**Version:** 1.0.0 · **Date:** 2026-08-20 · **Status:** Aligned with implementation (T2–T7)

This is the low-level design / API guide: every module, its public surface, and the
JSON contracts. Readers: implementers, the web app, and CI.

## 1. Package Layout

```
clinical-eval-harness/
├── data/golden/<doc_id>/        # source.md, golden.json, retrieval.json, raw_api.json
├── shared/
│   ├── schema.json               # single source of truth (docs, fields, eval config, aggregation)
│   └── prompts/{extract,judge}.txt
├── harness/
│   ├── __init__.py
│   ├── llm_client.py             # OpenAI-compatible chat client
│   ├── chunking.py               # structural + recursive chunking
│   ├── retrieval.py              # BM25 / dense / hybrid (RRF)
│   ├── retrieval_eval.py         # hit@k, recall@k, MRR
│   ├── extract.py                # LLM field extraction
│   ├── eval.py                   # rule-based + LLM-judge scoring
│   └── cli.py                    # subcommands
├── tests/                        # pytest (no network)
├── scripts/check_standards.py    # coding-standards gate
├── evals/                        # committed run artifacts
├── web/                          # Next.js viewer + /api routes
├── docs/                         # hld, lld, architecture, Decisions, standards
└── .github/workflows/ci.yml
```

## 2. harness/llm_client.py

```
class LLMClient:
    def __init__(self, base_url: str, api_key: str | None = None,
                 model: str, timeout_s: int = 60, max_retries: int = 3): ...
    def chat_json(self, system: str, user: str, temperature: float = 0.0) -> dict:
        """POST /chat/completions; parse the assistant content as JSON.
        Raises LLMError (non-retryable), LLMRateLimit (retryable), LLMJSONError."""
```

- Environment: `OLLAMA_API_KEY` / `OLLAMA_BASE_URL` (default https://ollama.com/v1).
- Retry with exponential backoff on 429/5xx; never retry 4xx.

## 3. harness/chunking.py

```
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section: str            # e.g. "Eligibility Criteria > Inclusion"
    text: str
    text_hash: str          # sha1, for dedupe/caching

def chunk_document(text: str, doc_id: str) -> list[Chunk]:
    """Structural (header-based) chunking with recursive fallback (800/100).
    Section path preserved as metadata for filtering + debug."""
```

## 4. harness/retrieval.py

```
@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    section: str
    text: str
    score: float
    strategy: str            # "bm25" | "dense" | "hybrid"

def build_index(docs: dict[str, str]) -> RetrievalIndex
class RetrievalIndex:
    def bm25(self, query: str, k: int = 5, doc_ids: list[str] | None = None) -> list[RetrievalResult]
    def dense(self, query: str, k: int = 5, doc_ids: list[str] | None = None) -> list[RetrievalResult]
    def hybrid(self, query: str, k: int = 5, rrf_k: int = 60, doc_ids: list[str] | None = None) -> list[RetrievalResult]
    def search_all(self, query: str, k: int = 5) -> dict[str, list[RetrievalResult]]
```
- Dense embeddings: local Ollama at `127.0.0.1:11434` (`/api/embed`), fallback
  `sentence-transformers/all-MiniLM-L6-v2`; embeddings cached to `evals/embeddings.jsonl`.

## 5. harness/retrieval_eval.py

```
def evaluate_retrieval(index: RetrievalIndex, gold: dict, k: int = 5) -> dict:
    """Per strategy: {hit_at_k, recall_at_k, mrr} per query + aggregate.
    gold = {queries: [{id, question, expected_span: [{section, quote}]}]}.
    A hit = any expected_span quote contained in the retrieved chunk text."""

def hit_at_k(results: list[RetrievalResult], gold_spans) -> float
def recall_at_k(...) -> float
def mrr(results, gold_spans) -> float
```

## 6. harness/extract.py

```
def extract_fields(doc_id: str, source_text: str, schema: dict,
                   client: LLMClient, prompt_path: str = "shared/prompts/extract.txt") -> dict:
    """Chunk source → prompt with schema → strict JSON per fields → {field: value}
    (null for unknown). Cached per (doc, model) at evals/runs/<model>.json."""
```

## 7. harness/eval.py

```
def score_field(field_key, category, gold, predicted, config) -> dict
    # numeric: tolerance_abs=1 / tolerance_pct=0.05
    # boolean: exact
    # date: fuzzy day
    # categorical: normalized match
    # free_text: llm_judge (prompt shared/prompts/judge.txt, judge_model)
    # list: per-item precision/recall
def evaluate_extraction(run: dict, gold: dict, config: dict, judge: LLMClient) -> dict
    # {precision, recall, f1} per field + aggregates
```

## 8. harness/cli.py -- command reference

```
python -m harness.cli extract --model glm-5.2 [--doc actt1]      # → evals/runs/<model>.json
python -m harness.cli eval --all-models                         # → evals/results.json, summary.json
python -m harness.cli retrieval-eval [--doc all]                # → evals/retrieval.json
python -m harness.cli eval --regression                         # exit 1 if below baseline
python -m harness.cli eval --mock-run                           # replay runs, no API (CI)
```

Exit codes: 0 success, 1 regression/failure. `--regression` compares current scores to
`evals/baseline.json` with tolerance 0.03 from `shared/schema.json#aggregation`.

## 9. Web API (web/, Next.js)

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Scorecard (served from `evals/summary.json` + `results.json`); empty state if absent |
| `/documents/:id` | GET | Golden doc: source.md, schema fields, retrieval queries |
| `/api/run` | POST | `{model}` → live extraction (server-side key) → fields vs gold |
| `/api/retrieval` | GET | Per-query what each strategy retrieved (debug view) |

All responses JSON; CORS for demo origin only; no auth (public demo), no secrets in client.

## 10. CI (GitHub Actions)

```
workflow: push + PR
jobs:
  standards: python setup → scripts/check_standards.py
  python:    setup 3.11 → pip install → pytest -q --cov harness → eval --mock-run → --regression
  web:       node 20 → cd web → npm ci → npm run build (continue-on-error: true)
```
No API keys in CI by construction (mock-run replays committed runs).

## 11. Data flow (happy path)

```
data/golden/*/source.md
      → chunking.py → chunks (with section metadata)
      → retrieval.py → BM25/dense/hybrid → retrieval_eval.py → evals/retrieval.json
      → extract.py (LLM) → evals/runs/<model>.json
      → eval.py (rules + judge) → evals/results.json + summary.json
      → cli.py --regression vs evals/baseline.json → CI gate
web/ renders evals/*.json (+ live /api/run)
```
