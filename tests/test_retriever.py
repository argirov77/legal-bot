import pytest
from unittest.mock import Mock

from app.retriever import Retriever


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_retrieve_returns_top_result():
    embedding_provider = Mock()
    embedding_provider.encode.return_value = [[0.1, 0.2, 0.3]]

    vectorstore = Mock()
    expected_results = [
        {"id": "doc-1", "text": "Top document", "meta": {"source": "a"}, "score": 0.95},
        {"id": "doc-2", "text": "Another document", "meta": {"source": "b"}, "score": 0.4},
    ]
    vectorstore.query.return_value = expected_results

    retriever = Retriever(vectorstore=vectorstore, embedding_provider=embedding_provider)

    results = await retriever.retrieve("session-123", "What is the law?", top_k=2)

    embedding_provider.encode.assert_called_once_with(["What is the law?"])
    vectorstore.query.assert_called_once_with(
        session_id="session-123", embedding=[0.1, 0.2, 0.3], top_k=2
    )
    assert results[0] == expected_results[0]
    assert len(results) == 2
