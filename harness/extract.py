"""LLM field extraction from clinical trial protocol text.

Builds a per-document prompt from shared/prompts/extract.txt plus the field
schema, calls the LLM, and returns a dict of field -> value (null for unknown).
Results are cached per (doc, model) at evals/runs/<model>.json.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .llm_client import LLMClient

GOLDEN_DIR = os.path.join("data", "golden")
SCHEMA_PATH = os.path.join("shared", "schema.json")
PROMPT_PATH = os.path.join("shared", "prompts", "extract.txt")
RUNS_DIR = os.path.join("evals", "runs")


def load_schema(path: str = SCHEMA_PATH) -> dict:
    """Load the shared schema (single source of truth)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_doc_schema(schema: dict, doc_id: str) -> dict:
    """Return the schema entry for one document."""
    for doc in schema.get("documents", []):
        if doc.get("doc_id") == doc_id:
            return doc
    raise KeyError(f"doc_id {doc_id} not in schema")


def get_golden_doc_ids() -> List[str]:
    """List doc_ids that have both source.md and golden.json."""
    if not os.path.isdir(GOLDEN_DIR):
        return []
    docs = []
    for name in sorted(os.listdir(GOLDEN_DIR)):
        src = os.path.join(GOLDEN_DIR, name, "source.md")
        gold = os.path.join(GOLDEN_DIR, name, "golden.json")
        if os.path.isfile(src) and os.path.isfile(gold):
            docs.append(name)
    return docs


def build_field_spec(fields: dict) -> str:
    """Render a compact, human-readable field spec for the prompt."""
    lines = []
    for key, cfg in fields.items():
        ftype = cfg.get("type", "string")
        desc = cfg.get("description", "")
        spec = f"- {key} ({ftype}): {desc}"
        if ftype == "categorical" and cfg.get("valid_values"):
            spec += f" [one of: {', '.join(cfg['valid_values'])}]"
        if ftype == "list":
            item = cfg.get("item_type", "string")
            spec += f" [list of {item}"
            if item == "categorical" and cfg.get("valid_values"):
                spec += f": {', '.join(cfg['valid_values'])}"
            if item == "object" and cfg.get("fields"):
                spec += f" with keys {sorted(cfg['fields'].keys())}"
            spec += "]"
        lines.append(spec)
    return "\n".join(lines)


def build_prompt(
    source_text: str,
    fields: dict,
    prompt_path: str = PROMPT_PATH,
) -> Tuple[str, str]:
    """Build (system, user) messages for extraction."""
    with open(prompt_path, "r", encoding="utf-8") as fh:
        system = fh.read()
    field_spec = build_field_spec(fields)
    user = (
        "Extract the following fields from the protocol text below.\n\n"
        f"FIELDS:\n{field_spec}\n\n"
        f"PROTOCOL TEXT:\n{source_text}\n\n"
        "Return a single JSON object with exactly these keys. "
        "Use null for any value not stated in the text."
    )
    return system, user


def extract_fields(
    doc_id: str,
    source_text: str,
    schema: dict,
    client: LLMClient,
    prompt_path: str = PROMPT_PATH,
) -> dict:
    """Extract all schema fields for one document via the LLM.

    Returns {field: value} with null for any field the model did not return.
    """
    doc = get_doc_schema(schema, doc_id)
    fields = doc["fields"]
    system, user = build_prompt(source_text, fields, prompt_path)
    result = client.chat_json(system, user)
    out: Dict[str, object] = {}
    for key in fields:
        out[key] = result.get(key)
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_filename(model: str) -> str:
    """Filesystem-safe filename for a model id (colons/slashes replaced)."""
    return model.replace(":", "-").replace("/", "-") + ".json"


def load_run(model: str) -> dict:
    """Load a committed run file, or an empty skeleton if absent."""
    path = os.path.join(RUNS_DIR, _run_filename(model))
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"model": model, "created_at": None, "extractions": {}}


def save_run(run: dict) -> str:
    """Write a run file, creating evals/runs/ as needed."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, _run_filename(run["model"]))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(run, fh, indent=2)
    return path


def list_run_models() -> List[str]:
    """List model ids that have committed run files."""
    if not os.path.isdir(RUNS_DIR):
        return []
    models = []
    for name in sorted(os.listdir(RUNS_DIR)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(RUNS_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("model"):
                models.append(data["model"])
        except (json.JSONDecodeError, OSError):
            continue
    return models


def run_extraction(
    model: str,
    doc_ids: List[str],
    schema: Optional[dict] = None,
    client: Optional[LLMClient] = None,
    prompt_path: str = PROMPT_PATH,
) -> dict:
    """Extract fields for the given docs and return a run dict.

    Merges into any existing run file for the same model so a single-doc
    re-run updates only that doc.
    """
    if schema is None:
        schema = load_schema()
    if client is None:
        client = LLMClient.from_env(model)

    run = load_run(model)
    for doc_id in doc_ids:
        src_path = os.path.join(GOLDEN_DIR, doc_id, "source.md")
        with open(src_path, "r", encoding="utf-8") as fh:
            source_text = fh.read()
        run["extractions"][doc_id] = extract_fields(
            doc_id, source_text, schema, client, prompt_path
        )
    run["model"] = model
    run["created_at"] = _now_iso()
    return run
