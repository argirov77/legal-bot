# Provider Adapter Design

## Goals

* Unify the way embeddings, LLM completions, and vector store operations are invoked.
* Make it easy to swap underlying services without touching business logic.
* Provide clear contracts for adapter implementations with minimal assumptions about the
  transport layer.

## Interface Summary

* `EmbeddingProvider.encode(texts: List[str]) -> List[List[float]]`
  * Batch-friendly method returning embeddings with the same order as the input.
  * Implementations may cache or retry as needed but must surface provider specific
    errors when unrecoverable.
* `LLMProvider.generate(prompt: str, max_tokens: int, **opts) -> str`
  * Simple text-in/text-out API with optional keyword configuration.
  * Keeps the orchestration layer agnostic to provider-specific SDKs.
* `VectorStoreAdapter.add(collection, ids, embeddings, metadatas, documents)`
  * Responsible for persisting vectors plus associated metadata.
  * MUST ensure idempotent inserts where possible.
* `VectorStoreAdapter.query(collection, query_embedding, k) -> results`
  * Returns at most `k` matches ordered by relevance score.
  * Result shape is adapter-defined but should include ids, scores, and metadata.

## Integration Points

* Embeddings pipeline will instantiate an `EmbeddingProvider` and remove direct SDK
  calls in `src/app/embeddings.py`.
* LLM invocation flow will reference an `LLMProvider` so orchestration logic can select
  providers dynamically.
* Vector store operations in `src/app/vectorstore.py` will delegate to a
  `VectorStoreAdapter` instance.

## Example Usage

```python
from app.providers import EmbeddingProvider, LLMProvider, VectorStoreAdapter

embedding_provider: EmbeddingProvider = get_embedding_provider()
vector_store: VectorStoreAdapter = get_vector_store_adapter()
llm: LLMProvider = get_llm_provider()

question = "What are the termination clauses?"
question_embedding = embedding_provider.encode([question])[0]

search_results = vector_store.query(
    collection="contracts",
    query_embedding=question_embedding,
    k=5,
)

context = "\n".join(hit.document for hit in search_results)
prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
answer = llm.generate(prompt, max_tokens=256, temperature=0.2)
```

The pseudocode above demonstrates how orchestration code can remain agnostic to concrete
implementations.
