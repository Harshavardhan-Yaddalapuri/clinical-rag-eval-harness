# Coding Standards — Clinical RAG Eval Harness

**Version:** 2.0.0 · **Enforced by:** `scripts/check_standards.py`, CI `standards` job,
orchestrator verification, and the `standards-watchdog` cron loop.

## 1. Scope
Every file committed to this repo: Python (`harness/`, `tests/`, `scripts/`),
TypeScript/Next.js (`web/`), YAML (CI), JSON (contracts, evals), Markdown (docs).
The bar is set by widely-adopted OSS projects: **ruff/pyflakes/pyright (FastAPI, dbt-core),
pytest + coverage (pandas, pytest itself), pre-commit (nearly every serious repo),
12-factor config (12factor.net), conventional CI (GitHub Actions reference patterns)**.

## 2. Structure & Naming
- **Directories:** `harness/` (Python engine), `web/` (Next.js app), `data/golden/<doc>/`
  (per-doc golden data), `shared/` (cross-cutting contract + prompts), `evals/` (committed
  run artifacts), `tests/` (pytest), `scripts/` (dev/verify tooling), `docs/` (HLD/LLD/ADR/architecture),
  `.github/workflows/` (CI).
- **Files:** `snake_case.py`; `kebab-case.tsx` / `PascalCase.tsx` per Next.js conventions;
  no spaces in any filename; config files lowercase (`schema.json`, `ci.yml`, `vercel.json`).
- **No scratch dirs, no temp files committed.** No `tmp/`, `scratch/`, `_old/`, `backup/`.
- **Exactly one repo root** — no sibling copy directories (checked by the standards script).

## 3. Python (harness/, tests/, scripts/)
- **Lint:** `ruff check` clean (E/F/W/I rules), `ruff format` applied.
- **Typing:** type hints on every public function signature; mypy (or pyright) clean
  where runnable; `Any` only with a documented justification.
- **Docstrings:** one-line summary + Args/Returns for public functions (Google style).
- **Logging:** module-level `logger = logging.getLogger(__name__)`; engine code logs,
  CLI entry points may print tables. Structured-ish logs (key=value), timestamps at
  INFO; never log secrets.
- **12-factor config:** config from env or `shared/schema.json`; no magic numbers inline;
  no credentials in code (dotenv pattern, `.env*` gitignored).
- **No inline interpreters** (`python3 -c`, `node -e`) in scripts/CI.
- **No em-dashes** in user-facing strings.

## 4. TypeScript / Next.js (web/)
- **Build:** `npm run build` MUST pass (ESLint + type-check via `next build`).
- **Typing:** strict TS; no `any` (exceptions documented); no unused vars.
- **Components:** Server Components by default; `"use client"` only for interaction.
  Props typed; no inline `style=` beyond spacing utilities.
- **Data access:** server-side `fs.readFile` of committed artifacts at request time —
  no build-time hardcoding of eval numbers, no client-side eval JSON imports.
- **No secrets client-side.** `OLLAMA_API_KEY` server-side only.
- **Responsive:** mobile-first; tables scroll on narrow screens.

## 5. YAML / JSON
- YAML: 2-space indent, no tabs, valid, no trailing whitespace.
- JSON: 2-space indent, valid, contract-checked (`shared/schema.json`, golden files,
  evals artifacts) — drift is a test failure, not a review comment.
- All artifacts under `data/` and `evals/` validated by tests.

## 6. Git & Repo Hygiene
- Meaningful commits (imperative, ≤72-char subject; body for non-trivial).
- No large binaries; no generated dirs (`.next/`, `node_modules/`, caches); no logs,
  `.DS_Store`, `__pycache__`.
- **CONTRIBUTING.md** (how to run/verify), **CHANGELOG.md** (Keep-a-Changelog format),
  **LICENSE** (MIT), **SECURITY.md** (reporting path), **CODEOWNERS** (if relevant).
- **pre-commit** config (ruff, trailing whitespace, secrets scan) in the repo.
- CI must stay green on every commit; broken CI is a P0.
- No force-push to shared branches.

## 7. Tests (the 30% bar)
- Unit tests: pytest, **no network** (mock LLM/embedding clients).
- Every eval metric + scoring rule tested (numeric tolerance, boolean, date, categorical,
  judge mock, RRF, hit@k/recall@k/MRR, null handling).
- **Coverage floor:** ≥80% on `harness/` (measured in CI; new features must not lower it).
- New feature → test in same PR; TDD where practical (test-first for eval semantics).
- CI: `pytest -q` + coverage + regression gate (`eval --mock-run` + `--regression`).

## 8. Documentation (docs-as-code)
- `docs/hld.md` (system), `docs/architecture.md` (scale), `docs/Decisions.md` (ADRs),
  `docs/lld.md` (module/API guide), `docs/standards.md` (this). Update on change.
- README: what/why/how + reproduce commands + real score table.
- ADRs record **why**, not just **what** (Decision, Context, Consequences).
- Changelog entries for every user-visible change.

## 9. Verification (the loop)
1. **Locally (every task):** worker runs `scripts/check_standards.py`; zero failures.
2. **CI:** `standards` job runs the same checker + `ruff check` + `pytest --cov` + `npm run build`
   on every push/PR; any failure blocks the merge.
3. **Orchestrator:** Arceus re-runs the checker + spot-audits before accepting "done"
   (workers' self-reports are never trusted).
4. **Watchdog:** cron `standards-watchdog` (every 6h, monitor script) alerts Telegram
   when the checker fails or CI is red — the loop that prompts Arceus to act.
5. **Reference OSS bar:** compare against FastAPI/pandas/ruff conventions when in doubt —
   if it would not merge in those repos, it does not merge here.

## 10. Definition of Done (a card is done ONLY when)
- [ ] Code exists in the canonical repo, committed, with real diff read.
- [ ] Verify command passes (unit tests / build / eval run).
- [ ] `scripts/check_standards.py` passes.
- [ ] No secrets, no em-dashes, no inline interpreters in the diff.
- [ ] Docs/ADRs/changelog updated if the change is user-visible.
- [ ] No side effects outside the repo (no rogue dirs, no deletes, no keys).
