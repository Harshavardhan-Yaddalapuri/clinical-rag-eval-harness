#!/usr/bin/env python
"""Verify the golden dataset: checks each golden dir has source.md + golden.json,
retrieval.json has 5 queries, schema valid JSON with 3 docs.
Prints a summary table (doc, #fields, #queries, source bytes)."""

import json
import os
import sys

GOLDEN_DIR = "/Users/harshavardhan/clinical-eval-harness/data/golden"
SCHEMA_PATH = "/Users/harshavardhan/clinical-eval-harness/shared/schema.json"
DOCS = ["actt1", "onc1", "area3"]

errors = []
warnings = []

def check_file_exists(path, label):
    if not os.path.isfile(path):
        errors.append(f"MISSING: {label} at {path}")
        return False
    return True

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f), None
    except json.JSONDecodeError as e:
        return None, f"INVALID JSON in {path}: {e}"
    except FileNotFoundError:
        return None, f"FILE NOT FOUND: {path}"

# 1. Check each golden directory
print("=" * 80)
print("GOLDEN DATASET VERIFICATION")
print("=" * 80)

summary_rows = []

for doc in DOCS:
    doc_dir = os.path.join(GOLDEN_DIR, doc)
    print(f"\n--- {doc} ---")
    
    row = {"doc": doc, "fields": 0, "queries": 0, "source_bytes": 0, "status": "OK"}
    
    # Check source.md exists and has content
    source_path = os.path.join(doc_dir, "source.md")
    if check_file_exists(source_path, f"{doc}/source.md"):
        source_size = os.path.getsize(source_path)
        row["source_bytes"] = source_size
        if source_size < 500:
            warnings.append(f"{doc}/source.md is suspiciously small ({source_size} bytes)")
        print(f"  source.md: {source_size} bytes")
    else:
        row["status"] = "FAIL"
    
    # Check golden.json exists and is valid JSON
    golden_path = os.path.join(doc_dir, "golden.json")
    golden_data, err = load_json(golden_path)
    if err:
        errors.append(err)
        row["status"] = "FAIL"
        print(f"  golden.json: {err}")
    else:
        # Count top-level fields (excluding doc_id which is just an identifier)
        field_count = len([k for k in golden_data.keys() if k != "doc_id"])
        row["fields"] = field_count
        print(f"  golden.json: {field_count} fields")
        
        # Verify key fields exist
        required_fields = [
            "nct_id", "brief_title", "lead_sponsor", "conditions", "phases",
            "study_type", "allocation", "enrollment_count", "overall_status",
            "start_date", "arms", "interventions", "eligibility_sex",
            "inclusion_criteria", "exclusion_criteria", "primary_outcomes",
            "secondary_outcomes"
        ]
        missing_fields = [f for f in required_fields if f not in golden_data]
        if missing_fields:
            warnings.append(f"{doc}/golden.json missing fields: {missing_fields}")
            print(f"  WARNING: missing fields: {missing_fields}")
        
        # Verify nct_id matches
        if golden_data.get("nct_id", "").startswith("NCT"):
            print(f"  nct_id: {golden_data['nct_id']}")
        else:
            errors.append(f"{doc}/golden.json has invalid nct_id: {golden_data.get('nct_id')}")
            row["status"] = "FAIL"
        
        # Verify enrollment is numeric
        ec = golden_data.get("enrollment_count")
        if not isinstance(ec, int):
            errors.append(f"{doc}/golden.json enrollment_count is not integer: {ec}")
            row["status"] = "FAIL"
        
        # Verify inclusion/exclusion are lists
        for field in ["inclusion_criteria", "exclusion_criteria", "conditions", "phases", "arms", "interventions", "primary_outcomes", "secondary_outcomes"]:
            val = golden_data.get(field)
            if not isinstance(val, list):
                errors.append(f"{doc}/golden.json '{field}' is not a list (got {type(val).__name__})")
                row["status"] = "FAIL"
        
        # Verify inclusion criteria is non-empty
        if not golden_data.get("inclusion_criteria"):
            errors.append(f"{doc}/golden.json inclusion_criteria is empty")
            row["status"] = "FAIL"
        if not golden_data.get("exclusion_criteria"):
            errors.append(f"{doc}/golden.json exclusion_criteria is empty")
            row["status"] = "FAIL"
    
    # Check retrieval.json exists, valid JSON, 5 queries
    retrieval_path = os.path.join(doc_dir, "retrieval.json")
    retrieval_data, err = load_json(retrieval_path)
    if err:
        errors.append(err)
        row["status"] = "FAIL"
        print(f"  retrieval.json: {err}")
    else:
        query_count = len(retrieval_data)
        row["queries"] = query_count
        if query_count != 5:
            errors.append(f"{doc}/retrieval.json has {query_count} queries, expected 5")
            row["status"] = "FAIL"
        else:
            print(f"  retrieval.json: {query_count} queries")
        
        # Verify each query has required fields
        for i, q in enumerate(retrieval_data):
            for field in ["id", "question", "expected_spans", "rationale"]:
                if field not in q:
                    errors.append(f"{doc}/retrieval.json query[{i}] missing field: {field}")
                    row["status"] = "FAIL"
            
            # Verify expected_spans is a non-empty list
            spans = q.get("expected_spans", [])
            if not isinstance(spans, list) or len(spans) < 2:
                warnings.append(f"{doc}/retrieval.json query[{i}] has < 2 expected_spans (cross-section synthesis required)")
                print(f"  WARNING: query[{i}] has {len(spans) if isinstance(spans, list) else 0} spans")
            else:
                # Verify each span has section and quote
                for j, span in enumerate(spans):
                    if "section" not in span or "quote" not in span:
                        errors.append(f"{doc}/retrieval.json query[{i}] span[{j}] missing section or quote")
                        row["status"] = "FAIL"
            
            # Verify question is non-empty
            if not q.get("question", "").strip():
                errors.append(f"{doc}/retrieval.json query[{i}] has empty question")
                row["status"] = "FAIL"
        
        # Print query IDs
        qids = [q.get("id", "?") for q in retrieval_data]
        print(f"  query IDs: {qids}")
    
    summary_rows.append(row)

# 2. Check schema.json
print(f"\n--- shared/schema.json ---")
schema_data, err = load_json(SCHEMA_PATH)
if err:
    errors.append(err)
    print(f"  {err}")
else:
    docs_in_schema = schema_data.get("documents", [])
    doc_count = len(docs_in_schema)
    print(f"  documents: {doc_count}")
    
    if doc_count != 3:
        errors.append(f"schema.json has {doc_count} documents, expected 3")
    else:
        # Verify each doc is in schema
        schema_doc_ids = [d.get("doc_id") for d in docs_in_schema]
        for doc in DOCS:
            if doc not in schema_doc_ids:
                errors.append(f"schema.json missing document: {doc}")
            else:
                # Check fields defined
                doc_entry = next(d for d in docs_in_schema if d.get("doc_id") == doc)
                fields = doc_entry.get("fields", {})
                field_count = len(fields)
                print(f"  {doc}: {field_count} field definitions")
                
                # Verify key field definitions exist
                required_field_defs = [
                    "nct_id", "lead_sponsor", "phases", "enrollment_count",
                    "arms", "inclusion_criteria", "exclusion_criteria",
                    "primary_outcomes", "masking"
                ]
                missing_defs = [f for f in required_field_defs if f not in fields]
                if missing_defs:
                    warnings.append(f"schema.json {doc} missing field definitions: {missing_defs}")
    
    # Check eval_config exists
    if "eval_config" not in schema_data:
        warnings.append("schema.json missing eval_config section")
    else:
        ec = schema_data["eval_config"]
        if "extraction" not in ec:
            warnings.append("schema.json missing extraction eval config")
        if "retrieval" not in ec:
            warnings.append("schema.json missing retrieval eval config")
        print(f"  eval_config: extraction={('extraction' in ec)}, retrieval={('retrieval' in ec)}")

# 3. Print summary table
print(f"\n{'=' * 80}")
print("SUMMARY TABLE")
print(f"{'=' * 80}")
print(f"{'Doc':<10} {'#Fields':<10} {'#Queries':<10} {'Source Bytes':<15} {'Status':<8}")
print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*15} {'-'*8}")
for row in summary_rows:
    print(f"{row['doc']:<10} {row['fields']:<10} {row['queries']:<10} {row['source_bytes']:<15} {row['status']:<8}")

# 4. Print errors and warnings
if warnings:
    print(f"\n--- WARNINGS ({len(warnings)}) ---")
    for w in warnings:
        print(f"  WARN: {w}")

if errors:
    print(f"\n--- ERRORS ({len(errors)}) ---")
    for e in errors:
        print(f"  FAIL: {e}")
    print(f"\nVERIFICATION FAILED: {len(errors)} error(s)")
    sys.exit(1)
else:
    total_fields = sum(r["fields"] for r in summary_rows)
    total_queries = sum(r["queries"] for r in summary_rows)
    total_bytes = sum(r["source_bytes"] for r in summary_rows)
    print(f"\nVERIFICATION PASSED")
    print(f"  3 docs, {total_fields} total fields, {total_queries} queries, {total_bytes} source bytes")
    if warnings:
        print(f"  {len(warnings)} warning(s) (non-blocking)")
    sys.exit(0)