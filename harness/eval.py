"""Rule-based + LLM-judge scoring for extraction results.

Scoring categories (mapped from schema field types):
  - numeric     : tolerance (abs / pct), field-level tolerance is authority
  - boolean     : exact
  - date        : fuzzy (year/month/day partial match)
  - categorical : normalized match
  - string      : normalized exact match
  - free_text   : LLM judge (shared/prompts/judge.txt)
  - list        : per-item precision / recall / f1

Aggregates are macro-averaged precision / recall / f1 per field, per document,
and across documents.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, List, Optional

from .llm_client import LLMClient
from .extract import load_run, list_run_models

GOLDEN_DIR = os.path.join("data", "golden")
JUDGE_PROMPT_PATH = os.path.join("shared", "prompts", "judge.txt")
RESULTS_PATH = os.path.join("evals", "results.json")
SUMMARY_PATH = os.path.join("evals", "summary.json")
BASELINE_PATH = os.path.join("evals", "baseline.json")

DEFAULT_TOLERANCE_ABS = 1
DEFAULT_TOLERANCE_PCT = 0.05
DEFAULT_REGRESSION_TOLERANCE = 0.03

TYPE_TO_CATEGORY = {
    "integer": "numeric",
    "number": "numeric",
    "float": "numeric",
    "boolean": "boolean",
    "date": "date",
    "categorical": "categorical",
    "list": "list",
    "free_text": "free_text",
    "string": "string",
}


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def _normalize_categorical(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _is_null(value: Any) -> bool:
    return value is None or value == "" or value == []


def _to_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _scalar_result(correct: bool, reason: str = "") -> dict:
    return {
        "correct": bool(correct),
        "precision": 1.0 if correct else 0.0,
        "recall": 1.0 if correct else 0.0,
        "f1": 1.0 if correct else 0.0,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Per-category scorers
# ---------------------------------------------------------------------------

def _score_numeric(gold: Any, predicted: Any, config: dict) -> dict:
    tol = config.get("tolerance", {})
    abs_tol = tol.get("abs", DEFAULT_TOLERANCE_ABS)
    pct_tol = tol.get("pct", DEFAULT_TOLERANCE_PCT)
    g = _to_number(gold)
    p = _to_number(predicted)
    if g is None:
        correct = _is_null(predicted)
    elif p is None:
        correct = False
    else:
        diff = abs(g - p)
        correct = diff <= abs_tol or diff <= pct_tol * abs(g)
    return _scalar_result(correct)


def _score_boolean(gold: Any, predicted: Any, config: dict) -> dict:
    g = _coerce_bool(gold)
    p = _coerce_bool(predicted)
    if g is None:
        correct = _is_null(predicted)
    else:
        correct = p == g
    return _scalar_result(correct)


def _normalize_date(value: Any) -> Optional[tuple]:
    if value is None or value == "":
        return None
    m = re.match(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?", str(value).strip())
    if not m:
        return None
    return (m.group(1), m.group(2), m.group(3))


def _date_fuzzy_match(g: tuple, p: tuple) -> bool:
    gy, gm, gd = g
    py, pm, pd = p
    if gy != py:
        return False
    if gm is None or pm is None:
        return True
    if gm != pm:
        return False
    if gd is None or pd is None:
        return True
    return gd == pd


def _score_date(gold: Any, predicted: Any, config: dict) -> dict:
    g = _normalize_date(gold)
    p = _normalize_date(predicted)
    if g is None:
        correct = _is_null(predicted)
    elif p is None:
        correct = False
    else:
        correct = _date_fuzzy_match(g, p)
    return _scalar_result(correct)


def _score_categorical(gold: Any, predicted: Any, config: dict) -> dict:
    g = _normalize_categorical(gold)
    p = _normalize_categorical(predicted)
    if g == "":
        correct = _is_null(predicted)
    else:
        correct = p == g
    return _scalar_result(correct)


def _score_string(gold: Any, predicted: Any, config: dict) -> dict:
    g = _normalize_text(gold)
    p = _normalize_text(predicted)
    if g == "":
        correct = _is_null(predicted)
    else:
        correct = p == g
    return _scalar_result(correct)


def _load_judge_prompt() -> str:
    with open(JUDGE_PROMPT_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


def _score_free_text(
    gold: Any, predicted: Any, config: dict, judge: Optional[LLMClient]
) -> dict:
    if judge is None:
        # No judge available (mock-run / CI): fall back to normalized exact.
        return _score_string(gold, predicted, config)
    if _is_null(gold):
        return _scalar_result(_is_null(predicted), reason="gold null")
    if _is_null(predicted):
        return _scalar_result(False, reason="predicted null")
    system = _load_judge_prompt()
    user = json.dumps(
        {
            "field": config.get("description", ""),
            "gold": gold,
            "candidate": predicted,
        }
    )
    verdict = judge.chat_json(system, user)
    correct = bool(verdict.get("correct", False))
    return _scalar_result(correct, reason=verdict.get("reason", ""))


def _item_key(item: Any, item_type: str, config: dict) -> str:
    if item_type == "object":
        fields = config.get("fields", {})
        for key in ("label", "name", "measure"):
            if key in fields and isinstance(item, dict) and key in item:
                return _normalize_text(item[key])
        if isinstance(item, dict):
            return _normalize_text(json.dumps(item, sort_keys=True))
        return _normalize_text(item)
    if item_type == "categorical":
        return _normalize_categorical(item)
    return _normalize_text(item)


def _match_list_items(gold: list, predicted: list, item_type: str, config: dict) -> int:
    pred_keys = [_item_key(x, item_type, config) for x in predicted]
    matched = 0
    used: set = set()
    for g_item in gold:
        gk = _item_key(g_item, item_type, config)
        for i, pk in enumerate(pred_keys):
            if i in used:
                continue
            if gk and pk and gk == pk:
                matched += 1
                used.add(i)
                break
    return matched


def _score_list(gold: Any, predicted: Any, config: dict) -> dict:
    g = gold if isinstance(gold, list) else []
    p = predicted if isinstance(predicted, list) else []
    if not g and not p:
        return {"correct": True, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    item_type = config.get("item_type", "string")
    matched = _match_list_items(g, p, item_type, config)
    precision = matched / len(p) if p else 0.0
    recall = matched / len(g) if g else 0.0
    f1 = _f1(precision, recall)
    return {
        "correct": precision == 1.0 and recall == 1.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Public scoring API
# ---------------------------------------------------------------------------

def score_field(
    field_key: str,
    category: str,
    gold: Any,
    predicted: Any,
    config: dict,
    judge: Optional[LLMClient] = None,
) -> dict:
    """Score one field. Returns {correct, precision, recall, f1, ...}."""
    if category == "numeric":
        result = _score_numeric(gold, predicted, config)
    elif category == "boolean":
        result = _score_boolean(gold, predicted, config)
    elif category == "date":
        result = _score_date(gold, predicted, config)
    elif category == "categorical":
        result = _score_categorical(gold, predicted, config)
    elif category == "list":
        result = _score_list(gold, predicted, config)
    elif category == "free_text":
        result = _score_free_text(gold, predicted, config, judge)
    else:
        result = _score_string(gold, predicted, config)
    result["field"] = field_key
    result["category"] = category
    result["gold"] = gold
    result["predicted"] = predicted
    return result


def _find_doc(schema: dict, doc_id: str) -> dict:
    for doc in schema.get("documents", []):
        if doc.get("doc_id") == doc_id:
            return doc
    raise KeyError(f"doc_id {doc_id} not in schema")


def _aggregate_fields(per_field: dict) -> dict:
    ps = [f["precision"] for f in per_field.values()]
    rs = [f["recall"] for f in per_field.values()]
    fs = [f["f1"] for f in per_field.values()]
    n = len(ps)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_fields": 0}
    return {
        "precision": round(sum(ps) / n, 4),
        "recall": round(sum(rs) / n, 4),
        "f1": round(sum(fs) / n, 4),
        "n_fields": n,
    }


def evaluate_extraction(
    run: dict,
    gold: dict,
    config: dict,
    judge: Optional[LLMClient] = None,
) -> dict:
    """Score one document's extraction against its golden values.

    run    : extracted {field: value} for one doc
    gold   : golden {field: value} for one doc (includes doc_id)
    config : the shared schema dict
    Returns {doc_id, per_field, aggregates}.
    """
    doc_id = gold.get("doc_id") or run.get("doc_id")
    if not doc_id:
        raise ValueError("gold and run both missing doc_id")
    doc = _find_doc(config, doc_id)
    fields = doc["fields"]
    per_field = {}
    for field_key, field_cfg in fields.items():
        category = TYPE_TO_CATEGORY.get(field_cfg.get("type", "string"), "string")
        per_field[field_key] = score_field(
            field_key,
            category,
            gold.get(field_key),
            run.get(field_key),
            field_cfg,
            judge=judge,
        )
    return {
        "doc_id": doc_id,
        "per_field": per_field,
        "aggregates": _aggregate_fields(per_field),
    }


def load_golden(doc_id: str) -> dict:
    path = os.path.join(GOLDEN_DIR, doc_id, "golden.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _aggregate_docs(per_document: dict) -> dict:
    ps, rs, fs = [], [], []
    for doc in per_document.values():
        agg = doc["aggregates"]
        ps.append(agg["precision"])
        rs.append(agg["recall"])
        fs.append(agg["f1"])
    n = len(ps)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_documents": 0}
    return {
        "precision": round(sum(ps) / n, 4),
        "recall": round(sum(rs) / n, 4),
        "f1": round(sum(fs) / n, 4),
        "n_documents": n,
    }


def evaluate_model(
    model: str,
    schema: dict,
    judge: Optional[LLMClient] = None,
) -> dict:
    """Score one model's committed run against all golden docs."""
    run = load_run(model)
    per_document = {}
    for doc_id, extraction in run.get("extractions", {}).items():
        gold = load_golden(doc_id)
        per_document[doc_id] = evaluate_extraction(extraction, gold, schema, judge=judge)
    return {
        "model": model,
        "per_document": per_document,
        "aggregates": _aggregate_docs(per_document),
    }


def evaluate_all_models(
    schema: dict,
    judge: Optional[LLMClient] = None,
    models: Optional[List[str]] = None,
) -> dict:
    """Score all committed runs. Returns {model: result}."""
    if models is None:
        models = list_run_models()
    return {model: evaluate_model(model, schema, judge=judge) for model in models}


def build_summary(results: dict) -> dict:
    """Compact scorecard: {model: aggregates}."""
    return {model: r["aggregates"] for model, r in results.items()}


def load_baseline(path: str = BASELINE_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def check_regression(
    summary: dict,
    baseline: dict,
    tolerance: float = DEFAULT_REGRESSION_TOLERANCE,
) -> dict:
    """Compare current summary to baseline. Returns per-model pass/fail report."""
    report = {}
    for model, cur in summary.items():
        base = baseline.get(model)
        if base is None:
            report[model] = {"pass": True, "note": "no baseline"}
            continue
        delta = cur["f1"] - base["f1"]
        report[model] = {
            "pass": delta >= -tolerance,
            "current_f1": cur["f1"],
            "baseline_f1": base["f1"],
            "delta": round(delta, 4),
        }
    return report


def write_results(results: dict, path: str = RESULTS_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return path


def write_summary(summary: dict, path: str = SUMMARY_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return path
