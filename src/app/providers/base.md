# Provider Interface Contracts

This document describes the abstract interfaces expected by the future adapter layer.
Implementations must follow these contracts so that downstream components can be swapped
without code changes.

## EmbeddingProvider

```
encode(texts: List[str]) -> List[List[float]]
```

Return a dense vector for each input text. Implementations SHOULD batch requests when
possible and MUST preserve the input ordering in the returned list.

* Inputs: a list of UTF-8 strings.
* Outputs: a list of embedding vectors where each vector is a list of floats of equal
  dimensionality.
* Error handling: raise a provider specific exception when remote services fail.

TODO: Wire concrete embedding providers through `src/app/embeddings.py`.

## LLMProvider

```
generate(prompt: str, max_tokens: int, **opts) -> str
```

Produce a completion for the given prompt. Implementations MUST respect `max_tokens`
limits and MAY use `**opts` for provider-specific tuning parameters (e.g. temperature,
stop sequences).

* Inputs: prompt text and max token budget.
* Outputs: a single string containing the model completion.
* Error handling: raise a provider specific exception when remote services fail.

TODO: Register LLM providers within the inference orchestration pipeline.

## VectorStoreAdapter

```
add(collection, ids, embeddings, metadatas, documents)
```

Insert embeddings into the backing vector store. The adapter is responsible for
normalizing metadata payloads and persisting both vectors and raw documents.

```
query(collection, query_embedding, k) -> results
```

Execute a similarity search returning at most `k` results ordered by relevance.

* Inputs: collection name, unique identifiers, embeddings, optional metadata, and raw
  documents for `add`; and query embedding plus `k` for `query`.
* Outputs: provider-defined objects representing insertion status or retrieved results.
* Error handling: raise a provider specific exception when the store cannot be reached
  or returns errors.

TODO: Integrate vector store adapters via `src/app/vectorstore.py`.
