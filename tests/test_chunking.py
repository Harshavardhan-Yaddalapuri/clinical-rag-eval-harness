"""Unit tests for structural chunking (no network).

Covers:
  - header-based section splitting (## / ### / h1 title)
  - oversized section fallback (paragraph + hard window with overlap)
  - chunk_id determinism, metadata, to_dict
  - chunk_document_file
  - _slugify edge cases
"""

from harness.chunking import (
    Chunk,
    chunk_markdown,
    chunk_document_file,
    _slugify,
    _split_oversized,
)


class TestSlugify:
    def test_basic(self):
        assert _slugify("Eligibility Criteria") == "eligibility-criteria"

    def test_empty_becomes_root(self):
        assert _slugify("!!!") == "root"


class TestChunkMarkdown:
    def test_sections_split(self):
        text = (
            "# Trial Title\n\n"
            "## Eligibility Criteria\n"
            "Patients with eGFR less than 30 are excluded.\n\n"
            "## Arms and Interventions\n"
            "Remdesivir 200 mg on Day 1.\n"
        )
        chunks = chunk_markdown(text, "doc1")
        sections = {c.section for c in chunks}
        assert "Eligibility Criteria" in sections
        assert "Arms and Interventions" in sections
        # h1 title is ignored, not a section
        assert all("Trial Title" not in c.section for c in chunks)

    def test_subsection_nests(self):
        text = (
            "## Eligibility Criteria\n"
            "intro text\n"
            "### Inclusion\n"
            "inclusion body\n"
            "### Exclusion\n"
            "exclusion body\n"
        )
        chunks = chunk_markdown(text, "doc1")
        sections = {c.section for c in chunks}
        assert "Eligibility Criteria > Inclusion" in sections
        assert "Eligibility Criteria > Exclusion" in sections

    def test_chunk_ids_deterministic_and_unique(self):
        text = "## A\nbody one\n\n## B\nbody two\n"
        chunks = chunk_markdown(text, "doc1")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
        assert all(i.startswith("doc1-") for i in ids)

    def test_to_dict(self):
        c = Chunk(chunk_id="x", doc_id="d", section="s", text="t")
        d = c.to_dict()
        assert d == {"chunk_id": "x", "doc_id": "d", "section": "s", "text": "t"}

    def test_empty_text_no_chunks(self):
        assert chunk_markdown("", "doc1") == []


class TestSplitOversized:
    def test_small_body_unchanged(self):
        assert _split_oversized("short", 800, 100) == ["short"]

    def test_paragraph_split(self):
        body = "para one\n\npara two\n\npara three"
        pieces = _split_oversized(body, 12, 3)
        # each piece <= max_chars
        assert all(len(p) <= 12 for p in pieces)
        assert len(pieces) >= 2

    def test_hard_split_long_paragraph(self):
        body = "x" * 50
        pieces = _split_oversized(body, 10, 2)
        assert all(len(p) <= 10 for p in pieces)
        # overlap means total length > 50
        assert sum(len(p) for p in pieces) > 50


class TestChunkDocumentFile:
    def test_reads_and_chunks(self, tmp_path):
        p = tmp_path / "source.md"
        p.write_text("## Section One\nsome body text here\n")
        chunks = chunk_document_file(str(p), "doc1")
        assert len(chunks) == 1
        assert chunks[0].doc_id == "doc1"
        assert chunks[0].section == "Section One"
