"""Unit tests for the extraction scoring engine (no network).

Covers:
  - numeric tolerance (abs / pct), field-level tolerance authority
  - boolean exact
  - date fuzzy
  - categorical normalized
  - string normalized
  - list per-item precision/recall/f1
  - free_text via MOCKED judge
  - per-field + document + model aggregates
  - regression check
"""

from harness.eval import (
    score_field,
    evaluate_extraction,
    check_regression,
    _normalize_categorical,
    _normalize_date,
    _date_fuzzy_match,
    _f1,
    DEFAULT_REGRESSION_TOLERANCE,
)
from harness.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Numeric
# ---------------------------------------------------------------------------

class TestNumeric:
    def test_exact_match(self):
        r = score_field("enrollment_count", "numeric", 1062, 1062, {})
        assert r["correct"] is True
        assert r["f1"] == 1.0

    def test_within_abs_tolerance(self):
        r = score_field("x", "numeric", 100, 101, {})
        assert r["correct"] is True

    def test_outside_tolerance(self):
        r = score_field("x", "numeric", 100, 110, {})
        assert r["correct"] is False

    def test_pct_tolerance_large_value(self):
        # 5% of 10000 = 500, so 10400 is within pct tolerance
        r = score_field("x", "numeric", 10000, 10400, {})
        assert r["correct"] is True

    def test_field_tolerance_is_authority(self):
        # enrollment_count has abs 0 / pct 0.0 -> exact only
        cfg = {"tolerance": {"abs": 0, "pct": 0.0}}
        r = score_field("enrollment_count", "numeric", 1062, 1063, cfg)
        assert r["correct"] is False
        r2 = score_field("enrollment_count", "numeric", 1062, 1062, cfg)
        assert r2["correct"] is True

    def test_string_number_coerced(self):
        r = score_field("x", "numeric", 1062, "1062", {})
        assert r["correct"] is True

    def test_gold_null_requires_null_prediction(self):
        r = score_field("x", "numeric", None, None, {})
        assert r["correct"] is True
        r2 = score_field("x", "numeric", None, 5, {})
        assert r2["correct"] is False


# ---------------------------------------------------------------------------
# Boolean
# ---------------------------------------------------------------------------

class TestBoolean:
    def test_exact_true(self):
        assert score_field("b", "boolean", True, True, {})["correct"] is True

    def test_exact_false(self):
        assert score_field("b", "boolean", False, False, {})["correct"] is True

    def test_mismatch(self):
        assert score_field("b", "boolean", True, False, {})["correct"] is False

    def test_string_coercion(self):
        assert score_field("b", "boolean", True, "true", {})["correct"] is True
        assert score_field("b", "boolean", False, "no", {})["correct"] is True

    def test_gold_null(self):
        assert score_field("b", "boolean", None, None, {})["correct"] is True
        assert score_field("b", "boolean", None, True, {})["correct"] is False


# ---------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------

class TestDate:
    def test_exact(self):
        assert score_field("d", "date", "2020-02-21", "2020-02-21", {})["correct"] is True

    def test_year_only_matches(self):
        assert score_field("d", "date", "2020-02-21", "2020", {})["correct"] is True

    def test_year_month_matches(self):
        assert score_field("d", "date", "2020-02-21", "2020-02", {})["correct"] is True

    def test_wrong_month(self):
        assert score_field("d", "date", "2020-02-21", "2020-03", {})["correct"] is False

    def test_wrong_day(self):
        assert score_field("d", "date", "2020-02-21", "2020-02-22", {})["correct"] is False

    def test_wrong_year(self):
        assert score_field("d", "date", "2020-02-21", "2021-02-21", {})["correct"] is False

    def test_gold_null(self):
        assert score_field("d", "date", None, None, {})["correct"] is True
        assert score_field("d", "date", None, "2020-02-21", {})["correct"] is False

    def test_normalize_date(self):
        assert _normalize_date("2020-02-21") == ("2020", "02", "21")
        assert _normalize_date("2020") == ("2020", None, None)
        assert _normalize_date(None) is None
        assert _normalize_date("") is None

    def test_date_fuzzy_match(self):
        assert _date_fuzzy_match(("2020", "02", "21"), ("2020", "02", "21"))
        assert _date_fuzzy_match(("2020", "02", "21"), ("2020", None, None))
        assert not _date_fuzzy_match(("2020", "02", "21"), ("2021", "02", "21"))


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------

class TestCategorical:
    def test_exact(self):
        assert score_field("c", "categorical", "NIH", "NIH", {})["correct"] is True

    def test_case_insensitive(self):
        assert score_field("c", "categorical", "NIH", "nih", {})["correct"] is True

    def test_underscore_vs_space(self):
        # normalization strips non-alphanumerics
        assert score_field("c", "categorical", "ACTIVE_NOT_RECRUITING",
                           "ACTIVE NOT RECRUITING", {})["correct"] is True

    def test_mismatch(self):
        assert score_field("c", "categorical", "NIH", "INDUSTRY", {})["correct"] is False

    def test_gold_null(self):
        assert score_field("c", "categorical", None, None, {})["correct"] is True
        assert score_field("c", "categorical", None, "NIH", {})["correct"] is False

    def test_normalize_categorical(self):
        assert _normalize_categorical("ACTIVE_NOT_RECRUITING") == "activenotrecruiting"
        assert _normalize_categorical("Phase 3") == "phase3"


# ---------------------------------------------------------------------------
# String
# ---------------------------------------------------------------------------

class TestString:
    def test_exact(self):
        assert score_field("s", "string", "hello", "hello", {})["correct"] is True

    def test_whitespace_collapse(self):
        assert score_field("s", "string", "a  b", "a b", {})["correct"] is True

    def test_mismatch(self):
        assert score_field("s", "string", "hello", "world", {})["correct"] is False


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

class TestList:
    def test_perfect(self):
        r = score_field("l", "list", ["a", "b"], ["a", "b"], {"item_type": "string"})
        assert r["precision"] == 1.0
        assert r["recall"] == 1.0
        assert r["f1"] == 1.0

    def test_partial(self):
        r = score_field("l", "list", ["a", "b"], ["a", "c"], {"item_type": "string"})
        assert r["precision"] == 0.5
        assert r["recall"] == 0.5
        assert r["f1"] == 0.5

    def test_empty_both(self):
        r = score_field("l", "list", [], [], {"item_type": "string"})
        assert r["correct"] is True
        assert r["f1"] == 1.0

    def test_gold_empty_predicted_nonempty(self):
        r = score_field("l", "list", [], ["a"], {"item_type": "string"})
        assert r["precision"] == 0.0
        assert r["recall"] == 0.0

    def test_categorical_items_normalized(self):
        r = score_field("l", "list", ["Phase 3"], ["phase3"],
                        {"item_type": "categorical"})
        assert r["recall"] == 1.0

    def test_object_items_match_on_label(self):
        cfg = {"item_type": "object", "fields": {"label": "string", "type": "categorical"}}
        gold = [{"label": "Remdesivir", "type": "DRUG"}]
        pred = [{"label": "Remdesivir", "type": "DRUG"}]
        r = score_field("l", "list", gold, pred, cfg)
        assert r["recall"] == 1.0


# ---------------------------------------------------------------------------
# Free text (judge MOCKED)
# ---------------------------------------------------------------------------

class TestFreeText:
    def test_judge_correct(self):
        judge = LLMClient(base_url="http://x", model="gpt-oss:20b")
        judge.chat_json = lambda s, u, temperature=0.0: {"correct": True, "reason": "ok"}
        r = score_field("f", "free_text", "gold text", "candidate text", {}, judge=judge)
        assert r["correct"] is True

    def test_judge_incorrect(self):
        judge = LLMClient(base_url="http://x", model="gpt-oss:20b")
        judge.chat_json = lambda s, u, temperature=0.0: {"correct": False, "reason": "no"}
        r = score_field("f", "free_text", "gold text", "candidate text", {}, judge=judge)
        assert r["correct"] is False

    def test_gold_null_short_circuits(self):
        judge = LLMClient(base_url="http://x", model="gpt-oss:20b")
        called = {"n": 0}

        def fake(s, u, temperature=0.0):
            called["n"] += 1
            return {"correct": True, "reason": "x"}

        judge.chat_json = fake
        r = score_field("f", "free_text", None, None, {}, judge=judge)
        assert r["correct"] is True
        assert called["n"] == 0  # judge not invoked for null gold

    def test_no_judge_falls_back_to_string(self):
        r = score_field("f", "free_text", "same", "same", {}, judge=None)
        assert r["correct"] is True
        r2 = score_field("f", "free_text", "same", "different", {}, judge=None)
        assert r2["correct"] is False


# ---------------------------------------------------------------------------
# Aggregates + regression
# ---------------------------------------------------------------------------

class TestAggregates:
    def _schema(self):
        return {
            "documents": [
                {
                    "doc_id": "actt1",
                    "fields": {
                        "nct_id": {"type": "string"},
                        "enrollment_count": {"type": "integer"},
                        "healthy_volunteers": {"type": "boolean"},
                    },
                }
            ]
        }

    def test_evaluate_extraction(self):
        gold = {
            "doc_id": "actt1",
            "nct_id": "NCT04280705",
            "enrollment_count": 1062,
            "healthy_volunteers": False,
        }
        run = {
            "nct_id": "NCT04280705",
            "enrollment_count": 1062,
            "healthy_volunteers": False,
        }
        result = evaluate_extraction(run, gold, self._schema())
        assert result["doc_id"] == "actt1"
        assert result["aggregates"]["f1"] == 1.0
        assert result["aggregates"]["n_fields"] == 3

    def test_evaluate_extraction_partial(self):
        gold = {
            "doc_id": "actt1",
            "nct_id": "NCT04280705",
            "enrollment_count": 1062,
            "healthy_volunteers": False,
        }
        run = {
            "nct_id": "NCT04280705",
            "enrollment_count": 9999,
            "healthy_volunteers": False,
        }
        result = evaluate_extraction(run, gold, self._schema())
        # 2 of 3 correct -> macro f1 = 2/3 (rounded to 4 decimals)
        assert abs(result["aggregates"]["f1"] - 0.6667) < 1e-4

    def test_f1_helper(self):
        assert _f1(1.0, 1.0) == 1.0
        assert _f1(0.5, 0.5) == 0.5
        assert _f1(0.0, 0.0) == 0.0


class TestRegression:
    def test_pass_within_tolerance(self):
        summary = {"glm-5.2": {"f1": 0.80}}
        baseline = {"glm-5.2": {"f1": 0.79}}
        report = check_regression(summary, baseline, DEFAULT_REGRESSION_TOLERANCE)
        assert report["glm-5.2"]["pass"] is True

    def test_fail_below_tolerance(self):
        summary = {"glm-5.2": {"f1": 0.70}}
        baseline = {"glm-5.2": {"f1": 0.80}}
        report = check_regression(summary, baseline, DEFAULT_REGRESSION_TOLERANCE)
        assert report["glm-5.2"]["pass"] is False

    def test_no_baseline_passes(self):
        summary = {"glm-5.2": {"f1": 0.80}}
        report = check_regression(summary, {}, DEFAULT_REGRESSION_TOLERANCE)
        assert report["glm-5.2"]["pass"] is True
        assert report["glm-5.2"]["note"] == "no baseline"
