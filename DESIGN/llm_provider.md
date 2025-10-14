# LLM Provider Integration Plan

## Purpose

Unify access to local large language models (BgGPT and Gemma) via a single `LLMProvider`
interface that can serve offline inference, local server delegates, and testing mocks.
This design outlines provider modes, configuration parameters, health-check routines,
and the integration points across the application (ingestion pipeline, RAG chain, API layer).

## Supported Operating Modes

| Mode | Description | Typical Usage |
| --- | --- | --- |
| `transformers` | Direct in-process loading of Hugging Face Transformers models. | Running BgGPT or Gemma on the same host as the FastAPI service. |
| `local_server` | Proxy to a locally deployed serving stack such as Text Generation Inference (TGI) or vLLM. | Offloading inference to a dedicated GPU worker accessible over HTTP/gRPC. |
| `mock` | Lightweight stub that returns canned responses for testing. | Unit tests, CI environments without model weights. |

The provider implementation must be able to switch between these modes based on runtime
configuration (environment variables, settings module, etc.).

## Loading Parameters

The following parameters must be supported irrespective of the mode; unsupported values
should surface clear errors:

- `model_path`: Filesystem path or model identifier for the primary checkpoint.
- `device_map`: Mapping used by Transformers to place layers onto devices (e.g., `auto`, `cuda:0`).
- `load_in_8bit`: Boolean flag enabling 8-bit quantised loading when supported.
- `torch_dtype`: Torch dtype override (e.g., `float16`, `bfloat16`).

Additional adapter-specific options (HTTP endpoints, authentication) can be layered on top,
but these core keys must remain consistent across configuration sources.

## Health-Check Contract

Every concrete provider must expose the following methods so the service can validate
availability before serving user traffic:

- `is_loaded() -> bool`: Returns `True` when the model weights (or remote endpoint) are ready.
- `mem_usage() -> dict[str, int | float]`: Surface memory metrics (GPU RAM, CPU RAM) for observability dashboards.
- `warmup() -> None`: Execute a low-cost prompt to populate caches and reduce initial latency.

These methods will be invoked by maintenance jobs and readiness probes. Failures should raise
provider-specific exceptions with actionable messages.

## Integration Points

| Area | File(s) | Notes |
| --- | --- | --- |
| Ingestion | `src/app/ingest/pipeline.py` | TODO hook for using BgGPT/Gemma to enrich chunks (summaries, metadata). |
| RAG Chain | `src/app/llm/rag_chain.py` | Planned orchestration of retrieval + generation using `LLMProvider`. |
| API | `src/app/main.py` | TODO placeholder to expose generation endpoints via FastAPI using provider dependency. |

Follow-up work will implement the provider class in `src/app/llm/llm_provider.py` and wire the
integration points listed above. Until then the TODO comments in those files indicate the touchpoints.

## Next Steps

1. Implement `LLMProvider` abstraction with factories for each mode.
2. Add dependency wiring (FastAPI dependency, background initialisation) using the health-check contract.
3. Replace TODO placeholders in ingestion, RAG chain, and API layers with concrete logic once
   BgGPT and Gemma adapters are available.
