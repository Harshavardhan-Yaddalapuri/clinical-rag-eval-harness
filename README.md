# Clinical RAG Eval Harness

Evaluation and regression harness for LLM-based clinical document pipelines. It measures
the two failure modes that matter in RAG (retrieval failure and reading failure, SANE
arXiv:2608.00658): how well retrieval finds the right spans, and how well extraction
turns documents into structured fields. Then it gates CI on it, so model swaps and
prompt changes cannot silently degrade extraction quality at live trial sites.

Built against real clinical trials from ClinicalTrials.gov. No PHI. No fabricated data.

## What it measures

| Layer | Question | Metrics |
|---|---|---|
| Retrieval | Does the pipeline recall the right spans for cross-section queries? | hit@k, recall@k, MRR (k=5) |
| Extraction | Does each model extract the golden-set fields correctly? | precision, recall, F1 per field |

Retrieval strategies compared: BM25 (lexical), dense (local embeddings), hybrid (RRF
fusion). Extraction models: `glm-5.2`, `deepseek-v4-pro:0813`, `qwen3.5:397b`, judged
by `gpt-oss:120b` for free-text fields.

## Real scores (from committed evals/)

Scores are real output of the harness against the golden set. The deepseek/qwen
identical scores were verified as genuine convergence: a fresh independent API call
reproduced the committed output byte-for-byte (temperature 0, verbatim fields).

| Model | Precision | Recall | F1 |
|---|---|---|---|
| glm-5.2 | 0.837 | 0.862 | 0.841 |
| deepseek-v4-pro:0813 | 0.826 | 0.851 | 0.830 |
| qwen3.5:397b | 0.826 | 0.851 | 0.830 |

### Retrieval (BM25 vs dense vs hybrid)

Real results on the 15 golden queries (evals/retrieval.json):

| Strategy | Hit@5 | Recall@5 | MRR |
|---|---|---|---|
| bm25 | 0.20 | 0.12 | 0.10 |
| dense | 0.40 | 0.19 | 0.26 |
| hybrid | 0.33 | 0.17 | 0.19 |

Honest note: on this small corpus dense beats hybrid and BM25, opposite of the common
assumption that hybrid fusion always wins. That is exactly why the harness exists:
strategy choices are measured, not assumed. Per-document breakdowns are in the committed
artifact.

## Golden set

Three real trials, hand-verified against ClinicalTrials.gov API v2 metadata:

- **actt1** NCT04280705 — Adaptive COVID-19 Treatment Trial (remdesivir)
- **onc1** NCT01642004 — CheckMate 017, nivolumab in squamous NSCLC
- **area3** NCT03036150 — DAPA-CKD, dapagliflozin in chronic kidney disease

Each has 31 extraction fields and 5 retrieval queries that require cross-section
synthesis (eligibility + labs + dosing, etc.). See `data/golden/<doc>/`.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# retrieval eval (BM25 + dense + hybrid on the golden queries)
python -m harness.cli retrieval-eval --doc all

# extraction + eval for one model
python -m harness.cli extract --model glm-5.2
python -m harness.cli eval --all-models

# CI regression gate (replays committed runs; no API keys)
python -m harness.cli eval --mock-run
python -m harness.cli eval --regression
```

## Repo layout

```
data/golden/<doc>/   source.md, golden.json, retrieval.json, raw_api.json
shared/              schema.json (single contract) + prompts
harness/             chunking, retrieval, retrieval_eval, extract, eval, cli
evals/               committed run artifacts + baseline
web/                 Next.js scorecard-first viewer + live /api/run
tests/               pytest (no network)
docs/                HLD, LLD, architecture, Decisions, standards
scripts/             check_standards.py (coding-standards gate)
.github/workflows/   ci.yml (standards + tests + regression gate + web build)
```

## Standards and quality

`scripts/check_standards.py` is the repo gate: Python lint heuristics, secrets scan,
JSON/YAML validity, git hygiene, file naming, one-repo-root enforcement. CI runs it on
every push. See `docs/standards.md` and `CONTRIBUTING.md`.

## Docs

- `docs/hld.md` — high-level design
- `docs/lld.md` — modules and API guide
- `docs/architecture.md` — scaling to millions of documents
- `docs/Decisions.md` — ADRs (why, not just what)
- `docs/standards.md` — coding standards + verification loop

## License

MIT.
