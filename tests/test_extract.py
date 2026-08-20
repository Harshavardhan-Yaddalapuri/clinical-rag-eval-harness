"""Unit tests for the extraction module (no network).

Covers:
  - build_field_spec / build_prompt rendering
  - extract_fields: JSON parse, null handling, missing keys -> null
  - run file load/save round-trip
  - get_golden_doc_ids against the real golden dir
"""

import os
from unittest.mock import patch

from harness.extract import (
    build_field_spec,
    build_prompt,
    extract_fields,
    load_run,
    save_run,
    list_run_models,
    get_golden_doc_ids,
    _run_filename,
)
from harness.llm_client import LLMClient


SAMPLE_FIELDS = {
    "nct_id": {"type": "string", "description": "NCT identifier"},
    "enrollment_count": {"type": "integer", "description": "Total enrollment"},
    "healthy_volunteers": {"type": "boolean", "description": "Healthy volunteers"},
    "sponsor_class": {
        "type": "categorical",
        "description": "Sponsor category",
        "valid_values": ["NIH", "INDUSTRY", "OTHER"],
    },
    "conditions": {"type": "list", "item_type": "string", "description": "Conditions"},
    "start_date": {"type": "date", "description": "Start date"},
}


class TestBuildFieldSpec:
    def test_renders_all_types(self):
        spec = build_field_spec(SAMPLE_FIELDS)
        assert "nct_id (string)" in spec
        assert "enrollment_count (integer)" in spec
        assert "healthy_volunteers (boolean)" in spec
        assert "one of: NIH, INDUSTRY, OTHER" in spec
        assert "list of string" in spec
        assert "start_date (date)" in spec

    def test_object_list_renders_keys(self):
        fields = {
            "arms": {
                "type": "list",
                "item_type": "object",
                "fields": {"label": "string", "type": "categorical"},
                "description": "Arms",
            }
        }
        spec = build_field_spec(fields)
        assert "with keys" in spec
        assert "label" in spec
        assert "type" in spec


class TestBuildPrompt:
    def test_system_and_user(self, tmp_path):
        prompt_path = tmp_path / "extract.txt"
        prompt_path.write_text("SYSTEM PROMPT HERE")
        system, user = build_prompt("some source text", SAMPLE_FIELDS, str(prompt_path))
        assert system == "SYSTEM PROMPT HERE"
        assert "some source text" in user
        assert "nct_id" in user
        assert "enrollment_count" in user


class TestExtractFields:
    def _schema(self):
        return {"documents": [{"doc_id": "actt1", "fields": SAMPLE_FIELDS}]}

    def test_returns_all_fields_with_nulls(self):
        client = LLMClient(base_url="http://x", model="m")
        client.chat_json = lambda s, u, temperature=0.0: {
            "nct_id": "NCT04280705",
            "enrollment_count": 1062,
            "healthy_volunteers": False,
            "sponsor_class": "NIH",
            "conditions": ["COVID-19"],
            "start_date": "2020-02-21",
        }
        out = extract_fields("actt1", "text", self._schema(), client)
        assert out["nct_id"] == "NCT04280705"
        assert out["enrollment_count"] == 1062
        assert out["healthy_volunteers"] is False
        assert out["sponsor_class"] == "NIH"
        assert out["conditions"] == ["COVID-19"]
        assert out["start_date"] == "2020-02-21"

    def test_missing_keys_become_null(self):
        client = LLMClient(base_url="http://x", model="m")
        client.chat_json = lambda s, u, temperature=0.0: {"nct_id": "NCT1"}
        out = extract_fields("actt1", "text", self._schema(), client)
        assert out["nct_id"] == "NCT1"
        assert out["enrollment_count"] is None
        assert out["healthy_volunteers"] is None
        assert out["sponsor_class"] is None
        assert out["conditions"] is None
        assert out["start_date"] is None

    def test_explicit_null_preserved(self):
        client = LLMClient(base_url="http://x", model="m")
        client.chat_json = lambda s, u, temperature=0.0: {
            "nct_id": None,
            "enrollment_count": None,
        }
        out = extract_fields("actt1", "text", self._schema(), client)
        assert out["nct_id"] is None
        assert out["enrollment_count"] is None

    def test_unknown_doc_raises(self):
        client = LLMClient(base_url="http://x", model="m")
        try:
            extract_fields("nope", "text", self._schema(), client)
            assert False, "expected KeyError"
        except KeyError:
            pass


class TestRunFiles:
    def test_run_filename_sanitizes(self):
        assert _run_filename("glm-5.2") == "glm-5.2.json"
        assert _run_filename("deepseek-v4-pro:0813") == "deepseek-v4-pro-0813.json"

    def test_load_run_missing_returns_skeleton(self, tmp_path):
        with patch("harness.extract.RUNS_DIR", str(tmp_path)):
            run = load_run("glm-5.2")
            assert run["model"] == "glm-5.2"
            assert run["extractions"] == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        run = {"model": "glm-5.2", "created_at": "x", "extractions": {"actt1": {"a": 1}}}
        with patch("harness.extract.RUNS_DIR", str(tmp_path)):
            path = save_run(run)
            assert os.path.exists(path)
            loaded = load_run("glm-5.2")
            assert loaded["extractions"]["actt1"]["a"] == 1

    def test_list_run_models(self, tmp_path):
        with patch("harness.extract.RUNS_DIR", str(tmp_path)):
            save_run({"model": "glm-5.2", "extractions": {}})
            save_run({"model": "qwen3.5:397b", "extractions": {}})
            models = list_run_models()
            assert "glm-5.2" in models
            assert "qwen3.5:397b" in models


class TestGoldenDocIds:
    def test_real_golden_docs_present(self):
        docs = get_golden_doc_ids()
        assert "actt1" in docs
        assert "onc1" in docs
        assert "area3" in docs
