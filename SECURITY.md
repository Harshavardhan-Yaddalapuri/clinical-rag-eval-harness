# Security Policy

## Reporting a vulnerability

Please do NOT open a public issue. Report privately to
harshavardhan.yaddalapuri@gmail.com with:

- A description of the issue
- Steps to reproduce
- Impact (what an attacker could do)
- Suggested fix (optional)

## Scope

- This repository is a public-data demo harness. It does not process PHI.
- The golden set is sourced from public ClinicalTrials.gov metadata. Do not add
  private/clinical data to this repo.
- API keys (e.g. OLLAMA_API_KEY) are server-side env vars only. Never commit them.
  The standards gate scans for leaked keys on every push.

## Response

I aim to acknowledge within 48 hours and triage within a week.

## Security-relevant facts (audit-friendly)

- Extraction judge and LLM calls: server-side only; no secrets in client code.
- CI runs with zero API keys (mock-run replays committed runs).
- Embedding cache and eval artifacts are committed for reproducibility; they contain
  no credentials.
