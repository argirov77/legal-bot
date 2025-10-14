"""Tests for the ingest pipeline using lightweight stubs."""
from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pytest
from starlette.datastructures import UploadFile

from app import ingest as ingest_module


@dataclass
class _DummyEmbedder:
    def encode(self, texts: List[str], show_progress_bar: bool = False) -> List[List[float]]:  # pragma: no cover - trivial
        return [[float(len(text))] for text in texts]


@dataclass
class _DummyCollection:
    name: str
    added: List[Dict[str, List[Any]]] = field(default_factory=list)

    def add(
        self,
        *,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, str]],
        ids: List[str],
    ) -> None:
        self.added.append(
            {
                "documents": list(documents),
                "embeddings": [list(item) for item in embeddings],
                "metadatas": list(metadatas),
                "ids": list(ids),
            }
        )


@dataclass
class _DummyChromaClient:
    collections: Dict[str, _DummyCollection] = field(default_factory=dict)
    persisted: bool = False

    def get_collection(self, name: str) -> _DummyCollection:
        if name not in self.collections:
            raise KeyError(name)
        return self.collections[name]

    def create_collection(self, name: str) -> _DummyCollection:
        collection = _DummyCollection(name=name)
        self.collections[name] = collection
        return collection

    def persist(self) -> None:
        self.persisted = True


@pytest.fixture()
def dummy_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _DummyChromaClient:
    dummy_client = _DummyChromaClient()
    monkeypatch.setattr(ingest_module, "embedder", _DummyEmbedder())
    monkeypatch.setattr(ingest_module, "chroma_client", dummy_client)
    monkeypatch.chdir(tmp_path)
    return dummy_client


def test_ingest_pipeline_persists_chunks(
    dummy_dependencies: _DummyChromaClient, tmp_path: Path
) -> None:
    session_id = "test-session"
    upload = UploadFile(filename="sample.txt", file=io.BytesIO(b"This is a sample contract text." * 5))

    body = asyncio.run(ingest_module.ingest(session_id, files=[upload]))
    assert body["status"] == "ok"
    assert body["added_chunks"] > 0
    assert dummy_dependencies.persisted is True

    collection_name = f"session_{session_id}"
    assert collection_name in dummy_dependencies.collections
    stored_batches = dummy_dependencies.collections[collection_name].added
    assert stored_batches, "Expected chunks to be stored in the dummy collection"
    first_batch = stored_batches[0]
    assert first_batch["documents"], "Stored batch should include documents"
    assert first_batch["metadatas"][0]["session_id"] == session_id
