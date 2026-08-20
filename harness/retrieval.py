"""Retrieval engine: BM25, dense, and hybrid (RRF) over chunked protocols.

No network access happens at import time. All backends are resolved lazily on
first use so the module stays importable and testable offline.

Strategies:
  - bm25   : lexical, rank-bm25 (Okapi BM25)
  - dense  : local Ollama embedding (127.0.0.1:11434) with a
             sentence-transformers/all-MiniLM-L6-v2 fallback
  - hybrid : reciprocal rank fusion (RRF, k=60) of bm25 + dense

Every strategy returns a list of dicts shaped
``{chunk_id, score, section, text}``. Metadata filtering (by doc_id or section
substring) is supported on all strategies.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk

RRF_K = 60
EMBED_CACHE_PATH = os.path.join("evals", "embeddings.jsonl")
OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "qwen3-embedding:0.6b"
FALLBACK_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return [t for t in text.lower().replace("-", " ").split() if t]


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """JSONL-backed cache of text -> embedding vector.

    Stored at evals/embeddings.jsonl so re-runs are fast and the cache can be
    committed or cleared independently of the code.
    """

    def __init__(self, path: str = EMBED_CACHE_PATH):
        self.path = path
        self._store: Dict[str, List[float]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._store[rec["h"]] = rec["v"]
                except (json.JSONDecodeError, KeyError):
                    continue

    def get(self, text: str) -> Optional[List[float]]:
        return self._store.get(_text_hash(text))

    def put(self, text: str, vector: List[float]) -> None:
        self._store[_text_hash(text)] = vector
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            for h, v in self._store.items():
                fh.write(json.dumps({"h": h, "v": v}) + "\n")
        self._dirty = False


class OllamaEmbedder:
    """Local Ollama embedding backend (127.0.0.1:11434)."""

    def __init__(self, base: str = OLLAMA_BASE, model: str = DEFAULT_EMBED_MODEL):
        self.base = base.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        try:
            import requests

            r = requests.get(f"{self.base}/api/tags", timeout=2)
            if r.status_code != 200:
                return False
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return self.model in names
        except Exception:
            return False

    def embed(self, texts: List[str]) -> List[List[float]]:
        import requests

        out: List[List[float]] = []
        for t in texts:
            r = requests.post(
                f"{self.base}/api/embed",
                json={"model": self.model, "input": t},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            emb = data.get("embeddings", [data.get("embedding")])[0]
            out.append([float(x) for x in emb])
        return out


class SentenceTransformerEmbedder:
    """Fallback dense backend via sentence-transformers/all-MiniLM-L6-v2."""

    def __init__(self, model: str = FALLBACK_EMBED_MODEL):
        self.model = model
        self._model = None

    def is_available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401

            return True
        except Exception:
            return False

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        vecs = model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


def _resolve_embedder() -> Optional[object]:
    """Return the first available embedder, or None if dense is unavailable."""
    ollama = OllamaEmbedder()
    if ollama.is_available():
        return ollama
    st = SentenceTransformerEmbedder()
    if st.is_available():
        return st
    return None


def _cosine(a: List[float], b: List[float]) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class Retriever:
    """Indexes chunks and serves bm25 / dense / hybrid queries."""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = list(chunks)
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: List[List[str]] = []
        self._embedder = None
        self._embedder_resolved = False
        self._cache = EmbeddingCache()
        self._build_bm25()

    # -- indexing ---------------------------------------------------------
    def _build_bm25(self) -> None:
        self._corpus = [_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._corpus)

    def _get_embedder(self) -> Optional[object]:
        if not self._embedder_resolved:
            self._embedder = _resolve_embedder()
            self._embedder_resolved = True
        return self._embedder

    # -- filtering --------------------------------------------------------
    def _filter_indices(
        self,
        doc_id: Optional[str] = None,
        section: Optional[str] = None,
    ) -> List[int]:
        idx = []
        for i, c in enumerate(self.chunks):
            if doc_id is not None and c.doc_id != doc_id:
                continue
            if section is not None and section.lower() not in c.section.lower():
                continue
            idx.append(i)
        return idx

    def _to_result(self, idx: int, score: float) -> dict:
        c = self.chunks[idx]
        return {
            "chunk_id": c.chunk_id,
            "score": score,
            "section": c.section,
            "text": c.text,
        }

    # -- strategies -------------------------------------------------------
    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        section: Optional[str] = None,
    ) -> List[dict]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        allowed = self._filter_indices(doc_id=doc_id, section=section)
        ranked = sorted(
            ((i, float(scores[i])) for i in allowed if scores[i] > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        return [self._to_result(i, s) for i, s in ranked[:top_k]]

    def search_dense(
        self,
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        section: Optional[str] = None,
    ) -> List[dict]:
        embedder = self._get_embedder()
        if embedder is None:
            return []
        q_vec = embedder.embed([query])[0]

        allowed = self._filter_indices(doc_id=doc_id, section=section)
        scored = []
        for i in allowed:
            c = self.chunks[i]
            c_vec = self._cache.get(c.text)
            if c_vec is None:
                c_vec = embedder.embed([c.text])[0]
                self._cache.put(c.text, c_vec)
            scored.append((i, _cosine(q_vec, c_vec)))
        self._cache.save()

        scored.sort(key=lambda x: x[1], reverse=True)
        return [self._to_result(i, s) for i, s in scored[:top_k]]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        section: Optional[str] = None,
    ) -> List[dict]:
        bm25 = self.search_bm25(query, top_k=top_k * 4, doc_id=doc_id, section=section)
        dense = self.search_dense(query, top_k=top_k * 4, doc_id=doc_id, section=section)

        rrf: Dict[str, float] = {}
        meta: Dict[str, dict] = {}
        for rank, r in enumerate(bm25):
            rrf[r["chunk_id"]] = rrf.get(r["chunk_id"], 0.0) + 1.0 / (RRF_K + rank + 1)
            meta[r["chunk_id"]] = r
        for rank, r in enumerate(dense):
            rrf[r["chunk_id"]] = rrf.get(r["chunk_id"], 0.0) + 1.0 / (RRF_K + rank + 1)
            meta.setdefault(r["chunk_id"], r)

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        out = []
        for chunk_id, score in ranked[:top_k]:
            r = dict(meta[chunk_id])
            r["score"] = score
            out.append(r)
        return out

    def search(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "hybrid",
        doc_id: Optional[str] = None,
        section: Optional[str] = None,
    ) -> List[dict]:
        if strategy == "bm25":
            return self.search_bm25(query, top_k=top_k, doc_id=doc_id, section=section)
        if strategy == "dense":
            return self.search_dense(query, top_k=top_k, doc_id=doc_id, section=section)
        if strategy == "hybrid":
            return self.search_hybrid(query, top_k=top_k, doc_id=doc_id, section=section)
        raise ValueError(f"unknown strategy: {strategy}")


def build_retriever(chunks: List[Chunk]) -> Retriever:
    """Convenience constructor."""
    return Retriever(chunks)
