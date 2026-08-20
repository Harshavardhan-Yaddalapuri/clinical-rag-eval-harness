#!/usr/bin/env python3
"""Coding-standards checker for the clinical-rag-eval-harness repo.

Stdlib-only (no third-party dependencies; PyYAML is used only when installed
and its absence is reported as a WARN, never a FAIL). Scans the repo, prints a
per-check PASS/FAIL report with a summary table, and exits 0 when no FAIL is
present, 1 otherwise. WARNings never fail the run.

Checks:
  1. Python lint (heuristic): py_compile syntax, unused imports (AST walk),
     undefined names (best-effort load/use), bare except clauses, and
     print() usage outside __main__ blocks in harness/ and tests/
     (logging required there).
  2. Secrets scan: regex over committed + working-tree files (skips .git,
     node_modules/, .next/, .env files, binaries) for AWS/OpenAI/Slack keys,
     private-key blocks, and OLLAMA_API_KEY literal assignments.
  3. JSON validity: every .json under data/, shared/, evals/ parses.
  4. YAML validity: every .yaml/.yml under .github/workflows/ parses; WARN
     when PyYAML is unavailable.
  5. Git hygiene: committed files matching forbidden patterns (node_modules/,
     .next/, *.pyc, .DS_Store, *.log, __pycache__/, .env, tmp/, scratch/,
     backup/, *_old/).
  6. File naming: snake_case.py for Python (except __init__.py/__main__.py),
     no spaces in any filename, no rogue sibling repo root.
  7. Docs presence: docs/standards.md required; README.md, docs/hld.md and
     docs/architecture.md advisory (WARN only).

Usage:
    python3 scripts/check_standards.py [REPO_ROOT]
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:  # optional deep YAML validation
    import yaml as _yaml  # noqa: F401

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

ROOT_DEFAULT = Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {".git", "node_modules", ".next", "__pycache__", ".venv", "venv", "out"}
TEXT_BYTE_LIMIT = 5 * 1024 * 1024

SECRET_PATTERNS = [
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "OpenAI-style secret key"),
    (re.compile(r"xox[baprs]-[0-9A-Za-z-]{10,}"), "Slack token"),
    (re.compile(r"BEGIN (RSA|OPENSSH|EC) PRIVATE KEY"), "private key material"),
    (re.compile(r"OLLAMA_API_KEY\s*=\s*[\"'][^\"']{4,}[\"']"), "OLLAMA_API_KEY literal assignment"),
]

FORBIDDEN_COMPONENTS = {"node_modules", ".next", "__pycache__", "tmp", "scratch", "backup"}
FORBIDDEN_EXACT = {".env", ".DS_Store"}
FORBIDDEN_SUFFIXES = (".pyc", ".log")
PY_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*\.py$")
PY_EXCEPTIONS = {"__init__.py", "__main__.py"}
MAGIC_NAMES = {
    "__name__", "__file__", "__doc__", "__all__", "__version__", "__path__",
    "__builtins__", "__annotations__", "__package__", "__loader__", "__spec__",
    "__cached__",
}
BUILTIN_NAMES = set(dir(builtins)) | MAGIC_NAMES


def _walk_files(root: Path, skip_env: bool = False):
    """Yield every file under root, pruning excluded directories."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fname in filenames:
            if skip_env and (fname == ".env" or fname.startswith(".env.")):
                continue
            yield Path(dirpath) / fname


def _tracked_files(root):
    """Return committed relative paths via git; None if git is unavailable."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return [p for p in proc.stdout.decode("utf-8", "replace").split("\0") if p]


# --------------------------------------------------------------------------- #
# 1. Python lint (heuristic, no external deps)                                 #
# --------------------------------------------------------------------------- #

def _import_bindings(tree):
    """Yield (lineno, bound_name) for every import statement."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.asname or alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                yield node.lineno, alias.asname or alias.name


def _unused_imports(tree):
    """Yield (lineno, bound_name) for imports never referenced in the file."""
    for lineno, bound in _import_bindings(tree):
        used = any(
            isinstance(node, ast.Name) and node.id == bound and node.lineno != lineno
            for node in ast.walk(tree)
        )
        if not used:
            yield lineno, bound


def _collect_target_names(target, defined):
    """Add every simple name bound inside a target expression."""
    if isinstance(target, ast.Name):
        defined.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _collect_target_names(elt, defined)
    elif isinstance(target, ast.Starred):
        _collect_target_names(target.value, defined)


def _collect_defined(tree):
    """Best-effort set of every name assigned/imported/declared in the file."""
    defined = set()

    def add_args(args):
        for group in (args.posonlyargs, args.args, args.kwonlyargs):
            for arg in group:
                defined.add(arg.arg)
        if args.vararg:
            defined.add(args.vararg.arg)
        if args.kwarg:
            defined.add(args.kwarg.arg)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
            add_args(node.args)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Lambda):
            add_args(node.args)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    defined.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                _collect_target_names(target, defined)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            _collect_target_names(node.target, defined)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            _collect_target_names(node.target, defined)
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars is not None:
                    _collect_target_names(item.optional_vars, defined)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                defined.add(node.name)
        elif isinstance(node, ast.Global):
            defined.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            defined.update(node.names)
        elif isinstance(node, ast.NamedExpr):
            _collect_target_names(node.target, defined)
        elif isinstance(node, ast.comprehension):
            _collect_target_names(node.target, defined)
        elif isinstance(node, ast.MatchAs) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.MatchStar) and node.name:
            defined.add(node.name)
    return defined


def _undefined_names(tree):
    """Yield (lineno, name) for Load names that are never defined anywhere."""
    known = _collect_defined(tree) | BUILTIN_NAMES
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in known and node.id not in seen:
                seen.add(node.id)
                yield node.lineno, node.id


def _bare_excepts(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            yield node.lineno


def _main_block_ranges(tree):
    """Line ranges covered by `if __name__ == \"__main__\":` blocks."""
    ranges = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            left = node.test.left
            if isinstance(left, ast.Name) and left.id == "__name__":
                ranges.append((node.lineno, node.end_lineno or node.lineno))
    return ranges


def _print_calls_outside_main(tree, main_ranges):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "print":
                if any(start <= node.lineno <= end for start, end in main_ranges):
                    continue
                yield node.lineno


def check_python(root):
    files = sorted(_walk_files(root))
    py_files = [path for path in files if path.suffix == ".py"]
    violations = []
    for path in py_files:
        rel = path.relative_to(root)
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            compile(source, str(rel), "exec")
        except SyntaxError as exc:
            violations.append(
                f"{rel}:{exc.lineno or 1}: syntax error: {exc.msg}"
            )
            continue
        tree = ast.parse(source, filename=str(rel))
        for lineno, name in _unused_imports(tree):
            violations.append(f"{rel}:{lineno}: unused import '{name}'")
        for lineno, name in _undefined_names(tree):
            violations.append(f"{rel}:{lineno}: undefined name '{name}'")
        for lineno in _bare_excepts(tree):
            violations.append(f"{rel}:{lineno}: bare except clause (name the exception type)")
        if str(rel).startswith(("harness/", "tests/")):
            main_ranges = _main_block_ranges(tree)
            for lineno in _print_calls_outside_main(tree, main_ranges):
                violations.append(
                    f"{rel}:{lineno}: use logging instead of print() outside __main__"
                )
    return len(py_files), violations


# --------------------------------------------------------------------------- #
# 2. Secrets scan                                                              #
# --------------------------------------------------------------------------- #
def check_secrets(root):
    scanned, hits = 0, []
    for path in _walk_files(root, skip_env=True):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == 0 or size > TEXT_BYTE_LIMIT:
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:8192]:  # binary
            continue
        text = raw.decode("utf-8", "replace")
        scanned += 1
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern, label in SECRET_PATTERNS:
                if pattern.search(line):
                    hits.append(f"{path.relative_to(root)}:{lineno}: possible {label}")
    return scanned, hits


# --------------------------------------------------------------------------- #
# 3. JSON validity                                                             #
# --------------------------------------------------------------------------- #
def check_json(root):
    checked, violations = 0, []
    for sub in ("data", "shared", "evals"):
        base = root / sub
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.json")):
            checked += 1
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                lineno = getattr(exc, "lineno", None) or 1
                violations.append(f"{path.relative_to(root)}:{lineno}: invalid JSON: {exc}")
    return checked, violations


# --------------------------------------------------------------------------- #
# 4. YAML validity (workflows)                                                 #
# --------------------------------------------------------------------------- #
def check_yaml(root):
    workflows = root / ".github" / "workflows"
    if not workflows.is_dir():
        return 0, [], False
    if not HAS_YAML:
        return 0, [], True  # warn: module unavailable, cannot deep-parse
    checked, violations = 0, []
    paths = sorted(workflows.glob("*.yaml")) + sorted(workflows.glob("*.yml"))
    for path in paths:
        checked += 1
        try:
            _yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, _yaml.YAMLError) as exc:
            mark = getattr(getattr(exc, "problem_mark", None), "line", None)
            lineno = mark + 1 if mark is not None else 1
            violations.append(f"{path.relative_to(root)}:{lineno}: invalid YAML: {exc}")
    return checked, violations, False


# --------------------------------------------------------------------------- #
# 5. Git hygiene (committed paths)                                             #
# --------------------------------------------------------------------------- #
def check_git_hygiene(tracked):
    violations = []
    for rel in tracked:
        parts = Path(rel).parts
        for i, part in enumerate(parts):
            base = parts[-1]
            if i < len(parts) - 1 and (part in FORBIDDEN_COMPONENTS or part.endswith("_old")):
                violations.append(f"{rel}: forbidden committed directory '{part}/'")
            if i == len(parts) - 1:
                if base in FORBIDDEN_EXACT or base.endswith(FORBIDDEN_SUFFIXES) or part.endswith("_old"):
                    violations.append(f"{rel}: forbidden committed file '{base}'")
    return violations


# --------------------------------------------------------------------------- #
# 6. File naming + duplicate roots                                             #
# --------------------------------------------------------------------------- #
def check_naming(root, tracked):
    violations = []
    for rel in tracked:
        parts = Path(rel).parts
        base = parts[-1]
        if any(" " in part for part in parts):
            violations.append(f"{rel}: filename contains spaces")
        if base.endswith(".py") and base not in PY_EXCEPTIONS and not PY_NAME_RE.match(base):
            violations.append(f"{rel}: python file must be snake_case (no caps/space)")
    rogue = []
    if root.parent.is_dir():
        for candidate in root.parent.iterdir():
            if not candidate.is_dir() or candidate == root:
                continue
            if re.search(r"^clinical[-_]*(?:rag[-_]*)?eval[-_]*harness$", candidate.name, re.IGNORECASE):
                rogue.append(candidate.name)
    if rogue:
        violations.append(
            f"rogue sibling dir present: {', '.join(rogue)} — exactly one repo root expected ({root})"
        )
    return violations


# --------------------------------------------------------------------------- #
# 7. Docs presence                                                             #
# --------------------------------------------------------------------------- #
def check_docs(root):
    violations, warns = [], []
    docs = root / "docs"
    if not (docs / "standards.md").is_file():
        violations.append("docs/standards.md:1: required docs/standards.md is missing")
    for name in ("hld.md", "architecture.md"):
        if not (docs / name).is_file():
            warns.append(f"docs/{name}: not present yet (advisory)")
    if not (root / "README.md").is_file():
        warns.append("README.md: not present yet (advisory)")
    return violations, warns


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #
def _line(label, status, detail):
    return f"  [{label}] {status}  {detail}"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Coding-standards checker (stdlib-only) for clinical-rag-eval-harness."
    )
    parser.add_argument("repo", nargs="?", default=str(ROOT_DEFAULT))
    args = parser.parse_args(argv)
    root = Path(args.repo).resolve()
    if not root.is_dir():
        print(f"FATAL: {root} is not a directory", file=sys.stderr)
        return 2

    print("=" * 78)
    print(f"CODING STANDARDS — {root}")
    print("=" * 78)

    rows = []      # (check, status, detail, violations, warns)
    all_violations, all_warns = [], []

    n_py, v = check_python(root)
    rows.append(("Python lint", "PASS" if not v else "FAIL", f"{n_py} file(s)", v))

    scanned, v = check_secrets(root)
    rows.append(("Secrets scan", "PASS" if not v else "FAIL", f"{scanned} file(s) scanned", v))

    n_json, v = check_json(root)
    rows.append(("JSON validity", "PASS" if not v else "FAIL", f"{n_json} file(s)", v))

    n_yaml, v, no_module = check_yaml(root)
    if no_module:
        rows.append(("YAML validity", "PASS", "no workflows present / yaml module unavailable", v))
        all_warns.append("PyYAML not installed — workflow YAML deep validation skipped")
    else:
        rows.append(("YAML validity", "PASS" if not v else "FAIL", f"{n_yaml} file(s)", v))

    tracked = _tracked_files(root)
    if tracked is None:
        tracked = [str(p.relative_to(root)) for p in _walk_files(root)]
        rows.append(("Git hygiene", "PASS", "git unavailable — fell back to working-tree scan", []))
    v = check_git_hygiene(tracked)
    rows.append(("Git hygiene", "PASS" if not v else "FAIL", f"{len(tracked)} committed path(s)", v))

    v = check_naming(root, tracked)
    rows.append(("File naming", "PASS" if not v else "FAIL", f"{len(tracked)} committed path(s)", v))

    v, w = check_docs(root)
    rows.append(("Docs presence", "PASS" if not v else "FAIL", "", v))
    all_warns.extend(w)

    for _, status, detail, v in rows:
        all_violations.extend(v)
        if status == "FAIL":
            for item in v:
                print(f"  FAIL  {item}")
        else:
            print(f"  PASS  {detail}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'CHECK':<16} {'STATUS':<6} {'DETAIL':<40}")
    print("-" * 62)
    for label, status, detail, _ in rows:
        print(f"{label:<16} {status:<6} {detail}")
    print("-" * 62)

    if all_warns:
        print(f"\nWARNINGS ({len(all_warns)})")
        for warn in all_warns:
            print(f"  WARN {warn}")
    if all_violations:
        print(f"\nFAILURES ({len(all_violations)})")
        for item in all_violations:
            print(f"  FAIL {item}")
        print("\nRESULT: FAIL — fix the violations above (exit 1)")
        return 1
    print(f"\nRESULT: PASS — no violations (exit 0)")
    if all_warns:
        print(f"         ({len(all_warns)} advisory warning(s) — non-blocking)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
