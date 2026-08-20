"""Structural chunking for clinical trial protocol markdown.

Splits a markdown protocol into retrieval-oriented chunks. Header-based
structural chunking is used first (## and ### headers define section paths),
with a recursive fallback that splits oversized sections by paragraph and then
by hard character window with overlap.

Chunks carry metadata (doc_id, section path) so the retrieval layer can filter
by section or document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

DEFAULT_MAX_CHARS = 800
DEFAULT_OVERLAP = 100

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    """A single retrieval chunk with provenance metadata."""

    chunk_id: str
    doc_id: str
    section: str
    text: str
    start: int = 0
    end: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section": self.section,
            "text": self.text,
        }


def _slugify(text: str) -> str:
    """Turn a section title into a stable, filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "root"


def _split_blocks(text: str) -> List[tuple]:
    """Split markdown into (section_path, body) blocks on ## and ### headers.

    A `#` (h1) line is treated as the document title and ignored for section
    purposes. `##` starts a new top-level section; `###` starts a subsection
    nested under the current top-level section.
    """
    lines = text.split("\n")
    blocks: List[tuple] = []
    path: List[str] = []
    buf: List[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            section = " > ".join(path) if path else "Document"
            blocks.append((section, body))

    for line in lines:
        m = _HEADER_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            if level == 1:
                # Document title: keep buffered content under current path.
                continue
            flush()
            if level == 2:
                path = [title]
            else:  # level >= 3
                if path:
                    path = path[:1] + [title]
                else:
                    path = [title]
            buf = []
        else:
            buf.append(line)
    flush()
    return blocks


def _split_oversized(body: str, max_chars: int, overlap: int) -> List[str]:
    """Recursive fallback: split by paragraph, then hard-split with overlap."""
    if len(body) <= max_chars:
        return [body]

    paragraphs = re.split(r"\n\s*\n", body)
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip() if current else para
        else:
            if current:
                chunks.append(current)
                current = ""
            # Hard-split an oversized paragraph with overlap.
            while len(para) > max_chars:
                chunks.append(para[:max_chars])
                para = para[max_chars - overlap:]
            current = para

    if current:
        chunks.append(current)
    return chunks


def chunk_markdown(
    text: str,
    doc_id: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Chunk]:
    """Chunk a markdown protocol into retrieval chunks with metadata.

    Returns a list of Chunk objects. chunk_id is deterministic and unique per
    document (``<doc_id>-<section_slug>-<index>``).
    """
    blocks = _split_blocks(text)
    chunks: List[Chunk] = []
    counter = 0

    for section, body in blocks:
        pieces = _split_oversized(body, max_chars, overlap)
        slug = _slugify(section)
        for piece in pieces:
            counter += 1
            chunk_id = f"{doc_id}-{slug}-{counter}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    section=section,
                    text=piece,
                )
            )
    return chunks


def chunk_document_file(
    path: str,
    doc_id: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Chunk]:
    """Read a markdown file and chunk it."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    return chunk_markdown(text, doc_id, max_chars=max_chars, overlap=overlap)
