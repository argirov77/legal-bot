from __future__ import annotations

import builtins

import pytest

from app import embeddings


@pytest.fixture(autouse=True)
def _reset_embedding_cache():
    embeddings.reset_embedding_model_cache()
    yield
    embeddings.reset_embedding_model_cache()


def test_embedding_model_uses_fallback_when_install_heavy_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSTALL_HEAVY", "false")

    real_import = builtins.__import__

    def _guarded_import(name: str, *args, **kwargs):
        if name == "sentence_transformers":
            raise AssertionError("sentence-transformers should not be imported when INSTALL_HEAVY is false")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    model = embeddings.EmbeddingModel()

    vectors = model.embed_texts(["hello", "world"])
    assert len(vectors) == 2
    assert all(len(vector) == embeddings.FALLBACK_DIMENSION for vector in vectors)
    assert vectors[0] != vectors[1]


def test_embedding_model_deterministic_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSTALL_HEAVY", "false")

    model_a = embeddings.EmbeddingModel()
    vectors_a = model_a.embed_texts(["same text"])

    embeddings.reset_embedding_model_cache()
    model_b = embeddings.EmbeddingModel()
    vectors_b = model_b.embed_texts(["same text"])

    assert vectors_a == vectors_b
