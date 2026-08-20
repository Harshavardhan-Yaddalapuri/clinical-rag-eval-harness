# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-08-20

### Added
- Golden corpus: 3 real clinical trials from ClinicalTrials.gov
  (ACTT-1 / CheckMate-017 / DAPA-CKD), 31 extraction fields and 15 synthesis
  retrieval queries per doc, hand-verified against the API metadata.
- Retrieval engine: BM25, dense (local embeddings), hybrid RRF with metadata
  filtering and embedding cache.
- Retrieval evaluation: hit@k, recall@k, MRR per strategy per document.
- Extraction engine: model-agnostic LLM extraction with null handling.
- Extraction evaluation: rule-based scoring (numeric/boolean/date/categorical)
  plus LLM-as-judge for free-text.
- CLI: extract, eval, retrieval-eval, regression, mock-run.
- CI: standards gate, pytest with coverage, regression gate, web build.
- Docs: HLD, LLD, architecture (scale), Decisions (ADRs), standards.
- Web viewer: Next.js scorecard-first UI with live `/api/run`.
- Deploy: Vercel project with server-side key.

### Fixed
- Duplicate repo root: consolidated to one canonical root enforced by the
  standards gate.
- Standards checker false positives: `# noqa` honored; CLI print exemption.

## [0.1.0] - 2026-08-20
- Initial scaffold: shared schema, extraction + judge prompts, ACTT-1 golden
  set seed.
