# Contributing

Thanks for touching this repo. It is a production-grade reference implementation,
so the bar is deliberate. Before you open a PR or push:

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

## Verify loop (run all three)
```bash
python scripts/check_standards.py   # coding-standards gate (must be zero FAIL)
pytest -q --cov=harness             # unit tests, no network
npm run build                       # web build (in web/)
```

## Working agreement
- One change per PR; read your own diff before requesting review.
- No secrets, no em-dashes in user-facing text, no inline interpreters
  (`python3 -c` / `node -e`) in scripts or CI.
- Every eval metric change ships with its test first (TDD for eval semantics).
- Update docs/ (HLD/LLD/architecture/Decisions) and CHANGELOG.md when behavior
  changes — stale docs are a P0.
- CI must stay green; broken CI is a P0.

## Regression gate
```bash
python -m harness.cli eval --mock-run   # replays committed runs, no API
python -m harness.cli eval --regression # exits 1 if below baseline
```
If you changed scoring semantics and the gate fails, you must commit a NEW
baseline with the change (never silently lower it).

## Reporting a vulnerability
See SECURITY.md.
