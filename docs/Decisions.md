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
to source. Volume is small (3 docs) -- the scale claim lives in docs/architecture.md.

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
(RRF fusion, k=60). The expected honest result was hybrid >= dense on synthesis queries
(best-practices survey, arXiv:2407.01219).
**Actual (committed, evals/retrieval.json, 15 queries):** dense beats hybrid beats BM25
(hit@k 0.40 / 0.33 / 0.20; MRR 0.26 / 0.19 / 0.10). On this small corpus the naive
assumption did not hold.
**Consequences:** Strategy choice is measured, not assumed. The README documents the real
numbers. A larger corpus / tuned RRF weight may change the ordering; the harness exists
to tell us when it does. This ADR is updated from evidence, not from vibes.

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

**Decision:** Demo deploys to Vercel (live URL a founder can click). No LocalStack --
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

## ADR-010: 80% coverage floor enforced in CI

**Status:** Accepted
**Context:** A job demo that ships with no tests or low coverage signals "prototype, not
product." Clinical AI pipelines have regulatory consequences; a regression harness that
cannot verify its own scoring logic is self-defeating. The initial coverage was 56%
(post-T4), which left scoring edge cases and CLI paths untested.
**Decision:** CI enforces `--cov-fail-under=80` on `harness/`. Coverage is measured on
every push, not just PRs. The floor is 80%, not 100%, because some code paths require
live API calls (extraction, judge) that CI cannot exercise without keys.
**Consequences:** Coverage rose from 56% to 89% (T7 added test_cli, test_retrieval,
test_llm_client, test_chunking). The 11% gap is the live-API surface (extract.py LLM
calls, judge paths), which is covered by the regression gate's mock-run replay instead.
The floor prevents silent coverage decay: a worker who adds untested code breaks CI.

## ADR-011: Scorecard-first web viewer

**Status:** Accepted
**Context:** A reviewer (founder, hiring manager) opens the link and decides in 10
seconds whether to keep reading. Leading with architecture prose or a login wall loses
that window. The artifact's value is the numbers, not the chrome.
**Decision:** The Next.js viewer (`web/`) renders the scorecard table as the first
screen (`GET /`). The table shows all models x precision/recall/F1 and all retrieval
strategies x hit@k/recall@k/MRR, pulled from committed `evals/summary.json` and
`evals/results.json`. No login, no auth wall, no "click here to see results." Document
detail (`GET /documents/:id`) and live extraction (`POST /api/run`) are one click below
the fold.
**Consequences:** Reviewer sees real numbers immediately. The viewer is a read-only
artifact by default (committed JSON), with an optional live extraction path for
reviewers who bring their own API key. Mobile-optimized for LinkedIn in-app browser.
