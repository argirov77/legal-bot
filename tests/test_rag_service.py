from __future__ import annotations

import asyncio
import os
from tempfile import SpooledTemporaryFile

from starlette.datastructures import Headers, UploadFile

from app.rag_service import MockLLM, RAGService


def _create_upload_file(content: str, filename: str = "test.txt") -> UploadFile:
    temp_file = SpooledTemporaryFile()
    temp_file.write(content.encode("utf-8"))
    temp_file.seek(0)
    headers = Headers({"content-type": "text/plain"})
    return UploadFile(file=temp_file, filename=filename, headers=headers)


def test_rag_service_ingest_and_answer() -> None:
    async def runner() -> None:
        os.environ["INSTALL_HEAVY"] = "true"
        service = RAGService(llm=MockLLM())
        sample_text = "First paragraph with important facts.\nSecond paragraph with more context."
        upload_file = _create_upload_file(sample_text, "sample.txt")

        stats = await service.ingest("session-1", [upload_file])

        assert stats["added_chunks"] > 0
        assert set(stats["durations"].keys()) == {"save", "extract", "chunk", "embed", "store"}

        response = await service.answer("session-1", "What facts are mentioned?", top_k=2, max_tokens=64)

        assert response["answer"].startswith("MOCK_ANSWER:")
        assert response["sources"], "Expected retrieved sources to be present"
        assert response["meta"]["retrieved_chunks"] == len(response["sources"])
        assert response["meta"]["max_tokens"] == 64

    try:
        asyncio.run(runner())
    finally:
        os.environ.pop("INSTALL_HEAVY", None)
