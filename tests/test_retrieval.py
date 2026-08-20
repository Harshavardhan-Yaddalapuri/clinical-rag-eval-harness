"""Unit tests for the retrieval engine (no network).

Covers:
  - tokenizer + text hash
  - EmbeddingCache load/get/put/save round-trip
  - OllamaEmbedder availability + embed (mocked requests)
  - SentenceTransformerEmbedder availability + embed (mocked)
  - _resolve_embedder precedence
  - _cosine math
  - Retriever: bm25, dense (mocked embedder), hybrid RRF, filtering, unknown strategy
"""

import os
from unittest.mock import patch, MagicMock

import numpy as np

from harness.retrieval import (
    _tokenize,
    _text_hash,
    EmbeddingCache,
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    _resolve_embedder,
    _cosine,
    Retriever,
    build_retriever,
)
from harness.chunking import Chunk


def _chunks():
    return [
        Chunk(chunk_id="A", doc_id="d1", section="Eligibility",
              text="remdesivir dosing 200 mg day 1 then 100 mg daily"),
        Chunk(chunk_id="B", doc_id="d1", section="Eligibility",
              text="eligibility criteria egfr less than 30 excluded"),
        Chunk(chunk_id="C", doc_id="d2", section="Outcomes",
              text="primary outcome mortality day 29"),
    ]


def _empty_cache(tmp_path):
    return EmbeddingCache(str(tmp_path / "emb.jsonl"))


class TestTokenize:
    def test_lowercase_and_split(self):
        assert _tokenize("Remdesivir-Dosing 200 mg") == ["remdesivir", "dosing", "200", "mg"]

    def test_empty(self):
        assert _tokenize("") == []


class TestTextHash:
    def test_deterministic(self):
        assert _text_hash("abc") == _text_hash("abc")
        assert _text_hash("abc") != _text_hash("abd")


class TestEmbeddingCache:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "emb.jsonl")
        cache = EmbeddingCache(path)
        assert cache.get("hello") is None
        cache.put("hello", [0.1, 0.2])
        cache.save()
        cache2 = EmbeddingCache(path)
        assert cache2.get("hello") == [0.1, 0.2]

    def test_load_skips_bad_lines(self, tmp_path):
        path = tmp_path / "emb.jsonl"
        good_h = _text_hash("good")
        path.write_text(
            f'{{"h": "{good_h}", "v": [1.0]}}\nnot json\n{{"bad": "shape"}}\n'
        )
        cache = EmbeddingCache(str(path))
        assert cache.get("good") == [1.0]

    def test_save_no_dirty_no_write(self, tmp_path):
        path = str(tmp_path / "emb.jsonl")
        cache = EmbeddingCache(path)
        cache.save()
        assert not os.path.exists(path)


class TestOllamaEmbedder:
    def test_is_available_true(self):
        emb = OllamaEmbedder(base="http://x", model="m")
        with patch("requests.get") as get:
            get.return_value.status_code = 200
            get.return_value.json.return_value = {"models": [{"name": "m"}]}
            assert emb.is_available() is True

    def test_is_available_false_wrong_model(self):
        emb = OllamaEmbedder(base="http://x", model="m")
        with patch("requests.get") as get:
            get.return_value.status_code = 200
            get.return_value.json.return_value = {"models": [{"name": "other"}]}
            assert emb.is_available() is False

    def test_is_available_false_on_error(self):
        emb = OllamaEmbedder(base="http://x", model="m")
        with patch("requests.get", side_effect=Exception("boom")):
            assert emb.is_available() is False

    def test_embed(self):
        emb = OllamaEmbedder(base="http://x", model="m")
        with patch("requests.post") as post:
            post.return_value.raise_for_status.return_value = None
            post.return_value.json.return_value = {"embeddings": [[0.5, 0.5]]}
            out = emb.embed(["hello"])
            assert out == [[0.5, 0.5]]

    def test_embed_single_embedding_key(self):
        emb = OllamaEmbedder(base="http://x", model="m")
        with patch("requests.post") as post:
            post.return_value.raise_for_status.return_value = None
            post.return_value.json.return_value = {"embedding": [0.25, 0.75]}
            out = emb.embed(["hello"])
            assert out == [[0.25, 0.75]]


class TestSentenceTransformerEmbedder:
    def test_is_available_true(self):
        emb = SentenceTransformerEmbedder()
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            assert emb.is_available() is True

    def test_is_available_false(self):
        emb = SentenceTransformerEmbedder()
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            assert emb.is_available() is False

    def test_embed(self):
        emb = SentenceTransformerEmbedder()
        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        emb._model = fake_model
        out = emb.embed(["a", "b"])
        assert out == [[0.1, 0.2], [0.3, 0.4]]


class TestResolveEmbedder:
    def test_ollama_preferred(self):
        with patch.object(OllamaEmbedder, "is_available", return_value=True):
            emb = _resolve_embedder()
            assert isinstance(emb, OllamaEmbedder)

    def test_fallback_to_sentence_transformer(self):
        with patch.object(OllamaEmbedder, "is_available", return_value=False), \
             patch.object(SentenceTransformerEmbedder, "is_available", return_value=True):
            emb = _resolve_embedder()
            assert isinstance(emb, SentenceTransformerEmbedder)

    def test_none_when_unavailable(self):
        with patch.object(OllamaEmbedder, "is_available", return_value=False), \
             patch.object(SentenceTransformerEmbedder, "is_available", return_value=False):
            assert _resolve_embedder() is None


class TestCosine:
    def test_identical(self):
        assert abs(_cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6

    def test_orthogonal(self):
        assert abs(_cosine([1.0, 0.0], [0.0, 1.0])) < 1e-6

    def test_zero_vector(self):
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestRetriever:
    def test_bm25_ranks_lexical(self):
        r = Retriever(_chunks())
        results = r.search_bm25("remdesivir dosing", top_k=5)
        assert results[0]["chunk_id"] == "A"

    def test_bm25_filters_by_doc(self):
        r = Retriever(_chunks())
        results = r.search_bm25("mortality", top_k=5, doc_id="d1")
        assert all(res["chunk_id"] != "C" for res in results)

    def test_bm25_filters_by_section(self):
        r = Retriever(_chunks())
        results = r.search_bm25("mortality", top_k=5, section="Outcomes")
        assert any(res["chunk_id"] == "C" for res in results)

    def test_dense_uses_embedder(self, tmp_path):
        r = Retriever(_chunks())
        r._cache = _empty_cache(tmp_path)
        fake = MagicMock()
        fake.embed.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
        r._embedder = fake
        r._embedder_resolved = True
        results = r.search_dense("anything", top_k=5)
        assert len(results) == 3

    def test_dense_no_embedder_returns_empty(self):
        r = Retriever(_chunks())
        r._embedder = None
        r._embedder_resolved = True
        assert r.search_dense("q") == []

    def test_hybrid_combines(self, tmp_path):
        r = Retriever(_chunks())
        r._cache = _empty_cache(tmp_path)
        fake = MagicMock()
        fake.embed.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
        r._embedder = fake
        r._embedder_resolved = True
        results = r.search_hybrid("remdesivir dosing", top_k=3)
        assert len(results) <= 3
        ids = [res["chunk_id"] for res in results]
        assert "A" in ids

    def test_search_dispatches(self, tmp_path):
        r = Retriever(_chunks())
        r._cache = _empty_cache(tmp_path)
        assert r.search("remdesivir", strategy="bm25")[0]["chunk_id"] == "A"
        # Force no embedder so dense is empty; hybrid then degrades to bm25-only
        r._embedder = None
        r._embedder_resolved = True
        assert r.search("remdesivir", strategy="dense") == []
        hybrid = r.search("remdesivir", strategy="hybrid")
        assert hybrid[0]["chunk_id"] == "A"  # bm25 result still surfaces

    def test_search_unknown_strategy_raises(self):
        r = Retriever(_chunks())
        try:
            r.search("q", strategy="nope")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_build_retriever(self):
        r = build_retriever(_chunks())
        assert isinstance(r, Retriever)
        assert len(r.chunks) == 3
