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

### Extraction (per-model aggregate)

| Model | Precision | Recall | F1 | Docs |
|---|---|---|---|---|
| glm-5.2 | 0.837 | 0.862 | 0.841 | 3 |
| deepseek-v4-pro:0813 | 0.826 | 0.851 | 0.830 | 3 |
| qwen3.5:397b | 0.826 | 0.851 | 0.830 | 3 |

### Extraction (per-model per-document, fields correct / 30)

| Model | actt1 | onc1 | area3 |
|---|---|---|---|
| glm-5.2 | 24/30 (0.800) | 25/30 (0.833) | 25/30 (0.833) |
| deepseek-v4-pro:0813 | 23/30 (0.767) | 25/30 (0.833) | 25/30 (0.833) |
| qwen3.5:397b | 23/30 (0.767) | 25/30 (0.833) | 25/30 (0.833) |

GLM-5.2 edges out the field on actt1 (the COVID-19 trial with the most complex
eligibility criteria and outcome lists). All three models converge on onc1 and area3.

### Retrieval (BM25 vs dense vs hybrid)

Real results on the 15 golden queries (evals/retrieval.json):

| Strategy | Hit@5 | Recall@5 | MRR |
|---|---|---|---|
| bm25 | 0.20 | 0.12 | 0.10 |
| dense | 0.40 | 0.19 | 0.26 |
| hybrid | 0.33 | 0.17 | 0.19 |

### Retrieval (per-strategy per-document)

| Strategy | Doc | Hit@5 | Recall@5 | MRR |
|---|---|---|---|---|
| bm25 | actt1 | 0.60 | 0.37 | 0.30 |
| bm25 | onc1 | 0.00 | 0.00 | 0.00 |
| bm25 | area3 | 0.00 | 0.00 | 0.00 |
| dense | actt1 | 0.80 | 0.43 | 0.67 |
| dense | onc1 | 0.20 | 0.07 | 0.07 |
| dense | area3 | 0.20 | 0.07 | 0.04 |
| hybrid | actt1 | 0.80 | 0.43 | 0.53 |
| hybrid | onc1 | 0.20 | 0.07 | 0.05 |
| hybrid | area3 | 0.00 | 0.00 | 0.00 |

Honest note: on this small corpus dense beats hybrid and BM25, opposite of the common
assumption that hybrid fusion always wins. That is exactly why the harness exists:
strategy choices are measured, not assumed. Per-document breakdowns show all strategies
struggle on onc1 and area3 (shorter, more dense protocol text with fewer distinct
sections to chunk), while actt1 (longer, more structured) is where retrieval works.
A larger corpus or tuned RRF weight may change the ordering; the harness exists to tell
us when it does. Per-query detail is in the committed artifact.

## Golden set

Three real trials, hand-verified against ClinicalTrials.gov API v2 metadata:

- **actt1** NCT04280705 -- Adaptive COVID-19 Treatment Trial (remdesivir)
- **onc1** NCT01642004 -- CheckMate 017, nivolumab in squamous NSCLC
- **area3** NCT03036150 -- DAPA-CKD, dapagliflozin in chronic kidney disease

Each has 30 extraction fields and 5 retrieval queries that require cross-section
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

### Full reproduction (real API calls)

```bash
# Source OLLAMA_API_KEY from env (never committed)
source ~/.hermes/.env

# Run all 3 extraction models against all 3 golden docs (9 API calls)
python -m harness.cli extract --model glm-5.2
python -m harness.cli extract --model deepseek-v4-pro:0813
python -m harness.cli extract --model qwen3.5:397b

# Score all models + write results/summary/baseline
python -m harness.cli eval --all-models

# Retrieval eval (local embeddings via Ollama 127.0.0.1:11434)
python -m harness.cli retrieval-eval --doc all
```

### CI pipeline

```bash
# What CI runs on every push:
python scripts/check_standards.py     # coding-standards gate
pytest -q --cov=harness --cov-fail-under=80 tests/  # unit tests, 89% coverage
python -m harness.cli eval --mock-run  # replay committed runs, no API keys
python -m harness.cli eval --regression  # compare vs baseline (tolerance 0.03)
cd web && npm ci && npm run build       # Next.js viewer build
```

## Repo layout

```
data/golden/<doc>/   source.md, golden.json, retrieval.json, raw_api.json
shared/              schema.json (single contract) + prompts
harness/             chunking, retrieval, retrieval_eval, extract, eval, cli
evals/               committed run artifacts + baseline
web/                 Next.js scorecard-first viewer + live /api/run (reads a committed
                     snapshot of evals/data/shared under web/; refresh with
                     `node web/scripts/sync-snapshot.mjs` after re-running evals)
tests/               pytest (no network, 160 tests, 89% coverage)
docs/                HLD, LLD, architecture, Decisions, standards
scripts/             check_standards.py (coding-standards gate)
.github/workflows/   ci.yml (standards + tests + regression gate + web build)
```

## Architecture decisions (summary)

| ADR | Decision | Why |
|---|---|---|
| 001 | Golden set from real ClinicalTrials.gov data | Reproducible, no PHI, values traceable to source |
| 002 | Eval harness for BOTH retrieval and extraction | Extraction-only catches half the RAG failure modes |
| 003 | Hybrid RRF as reference, measured not assumed | Dense actually wins on this corpus; harness tells us when that changes |
| 004 | LLM-as-judge only for free-text; deterministic rules otherwise | Cheap, auditable, deterministic for 28/30 field types |
| 005 | Regression gate replays committed runs in CI | Zero API keys in CI; catches scoring-semantic regressions |
| 006 | Vercel, not LocalStack | Live URL a founder can click; no fake AWS |
| 007 | One canonical repo root | Workers never split writes; standards gate enforces |
| 008 | Model set locked for extraction eval | Honest 3-model comparison; pluggability is a feature, not a claim |
| 009 | Embedded fallback for dense retrieval | Zero API cost; reproducible; CI/tests stay network-free |
| 010 | 80% coverage floor in CI | Prevents test debt from accumulating silently |
| 011 | Scorecard-first web viewer | Reviewer sees the numbers before the narrative |

See `docs/Decisions.md` for full ADRs with context and consequences.

## Standards and quality

`scripts/check_standards.py` is the repo gate: Python lint heuristics, secrets scan,
JSON/YAML validity, git hygiene, file naming, one-repo-root enforcement. CI runs it on
every push. See `docs/standards.md` and `CONTRIBUTING.md`.

## Docs

- `docs/hld.md` -- high-level design
- `docs/lld.md` -- modules and API guide
- `docs/architecture.md` -- scaling to millions of documents (production design)
- `docs/Decisions.md` -- ADRs (why, not just what)
- `docs/standards.md` -- coding standards + verification loop

## License

MIT.