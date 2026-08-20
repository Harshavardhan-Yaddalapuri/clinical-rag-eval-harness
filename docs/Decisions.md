# Decisions (ADRs)

Why this repo exists and the choices behind it. Written for a reviewer who wants to
understand architectural thinking, not just see a demo. Format: Context → Decision → Consequences.

## ADR-001: Golden set from real ClinicalTrials.gov data

**Status:** Accepted
**Context:** A job demo for a stealth clinical-research AI startup. The founder reads
messages; the artifact must be honest and verifiable. Fabricated data would poison the
"trust the numbers" story.
**Decision:** Golden set = 3 real public trials from ClinicalTrials.gov API v2:
ACTT-1 (NCT04280705, COVID-19), CheckMate-017 (NCT01642004, oncology/NSCLC),
DAPA-CKD (NCT03036150, nephrology/CKD). 31 extraction fields + 5 synthesis retrieval
queries per doc, hand-verified against the API metadata. No PHI.
**Consequences:** Reproducible by any reviewer; no PHI/NDA risk; values are traceable
to source. Volume is small (3 docs) — the scale claim lives in docs/architecture.md.

## ADR-002: Eval harness for BOTH retrieval and extraction

**Status:** Accepted
**Context:** RAG has two failure modes (SANE, arXiv:2608.00658): retrieval failure
(right span never recalled) and reading failure (right span, wrong answer). Most demo
pipelines only show extraction; an extraction-only harness catches half the failures.
The JD's core is "retrieval-augmented pipelines for document parsing and extraction."
**Decision:** The harness measures retrieval quality (BM25 vs dense vs hybrid, hit@k /
recall@k / MRR) AND extraction quality (per-field scoring across models). The CI gate
guards both.
**Consequences:** Larger golden set (queries + fields); honest comparison across
strategies; the README claims are grounded in the best-practices survey
(arXiv:2407.01219).

## ADR-003: Hybrid RRF over BM25 + dense as reference strategy

**Decision:** Retrieval strategies: BM25 (sparse), dense (local embeddings), hybrid
(RRF fusion, k=60). Expected honest result: hybrid >= dense on synthesis queries;
BM25 wins lexical. This matches the best-practices survey and is measurable here.

## ADR-004: LLM-as-judge only for free-text; deterministic rules otherwise

**Decision:** Numeric fields score with tolerance (abs 1 / pct 5%), booleans exact,
dates fuzzy-day, categoricals normalized. Free-text fields use a strict judge prompt
(gpt-oss) with null-aware rules. Judge outputs are one JSON object, no prose.
**Consequences:** Deterministic, cheap, auditable scoring; LLM judgment confined to
the genuinely semantic field type.

## ADR-005: Regression gate replays committed runs in CI

**Decision:** CI gate runs `eval --mock-run` (replay committed runs) then
`--regression` (compare vs committed baseline, tolerance 0.03). Zero API keys in CI.
**Consequences:** CI is deterministic and free; the gate catches regressions in the
scoring semantics and in committed run artifacts, not live model drift (which is a
nightly job in production).

## ADR-006: Vercel, not LocalStack, for deployment

**Decision:** Demo deploys to Vercel (live URL a founder can click). No LocalStack —
a fake AWS costs time and adds zero reviewer-visible value.
**Consequences:** Fast, free, zero-login mobile page. The scale architecture is
documented, not emulated.

## ADR-007: One canonical repo root (board workspace)

**Decision:** The kanban board workspace (`~/clinical-rag-eval-harness`) is the single
canonical repo. A duplicate root is a standards failure (enforced by
scripts/check_standards.py).
**Consequences:** Workers never split writes across roots; the standards gate catches
rogue siblings.

## ADR-008: Model set locked for extraction eval

**Decision:** Extractors: `glm-5.2`, `deepseek-v4-pro:0813`, `qwen3.5:397b`; judge
`gpt-oss:120b`. More models (minimax, kimi, nemotron) are runnable via `--model`
pluggability but are not in the committed scorecard.
**Consequences:** Honest 3-model comparison; pluggability is a feature, not a claim.

## ADR-009: Embedded fallback for dense retrieval

**Decision:** Dense embeddings via local Ollama (`127.0.0.1:11434`,
`qwen3-embedding:0.6b`) with a `sentence-transformers/all-MiniLM-L6-v2` fallback;
cache at `evals/embeddings.jsonl`.
**Consequences:** Zero API cost; reproducible; the fallback keeps CI/tests network-free.
