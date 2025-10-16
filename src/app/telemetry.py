"""Centralised observability helpers for structured lifecycle logging."""

from __future__ import annotations
import json
import logging
import os
import platform
import socket
import stat
import subprocess
import sys
import time
import traceback
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


LOGGER = logging.getLogger("app.telemetry")

_ENV_KEYS_TO_LOG: tuple[str, ...] = (
    "LLM_MODEL_PATH",
    "LLM_BG1_PATH",
    "LLM_BG2_PATH",
    "LLM_PROVIDER",
    "INSTALL_HEAVY",
    "FORCE_LOAD_ON_START",
    "USE_GPU",
    "CHROMA_PERSIST_DIR",
    "OCR_LANG",
    "LLM_DEVICE",
    "LLM_DEVICE_MAP",
    "LLM_TORCH_DTYPE",
    "LLM_QUANT",
    "LLM_MAX_TOKENS",
    "LLM_TEMPERATURE",
    "EMBEDDING_MODEL_PATH",
    "EMBEDDING_DEVICE",
    "VECTOR_STORE",
)


@dataclass(slots=True)
class NvidiaSMIRecord:
    index: int
    name: str
    memory_total_mb: Optional[int] = None
    memory_used_mb: Optional[int] = None
    utilisation_gpu: Optional[int] = None


def _safe_getuid() -> Optional[int]:  # pragma: no cover - platform dependent
    try:
        return os.getuid()  # type: ignore[attr-defined]
    except AttributeError:
        return None


def _safe_getgid() -> Optional[int]:  # pragma: no cover - platform dependent
    try:
        return os.getgid()  # type: ignore[attr-defined]
    except AttributeError:
        return None


def _run_command(command: list[str], *, timeout: float = 5.0) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as error:  # pragma: no cover - depends on runtime
        return 1, "", str(error)
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    return completed.returncode, stdout, stderr


def _parse_nvidia_smi(output: str) -> list[NvidiaSMIRecord]:
    records: list[NvidiaSMIRecord] = []
    for line in output.splitlines():
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if not parts:
            continue
        try:
            index = int(parts[0])
        except (TypeError, ValueError):
            continue
        name = parts[1] if len(parts) > 1 else "GPU"
        total = _try_parse_int(parts[2]) if len(parts) > 2 else None
        used = _try_parse_int(parts[3]) if len(parts) > 3 else None
        util = _try_parse_int(parts[4]) if len(parts) > 4 else None
        records.append(
            NvidiaSMIRecord(
                index=index,
                name=name,
                memory_total_mb=total,
                memory_used_mb=used,
                utilisation_gpu=util,
            )
        )
    return records


def _capture_nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    returncode, stdout, stderr = _run_command(command)
    records = None
    if returncode == 0 and stdout:
        records = [record.__dict__ for record in _parse_nvidia_smi(stdout)]
    return {
        "command": command,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "records": records,
    }


def _try_parse_int(value: str | None) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _format_exception(error: BaseException) -> str:
    return "".join(traceback.format_exception(error.__class__, error, error.__traceback__))


def log_event(
    logger: Optional[logging.Logger],
    step: str,
    *,
    level: str = "info",
    trace_id: str | None = None,
    req_id: str | None = None,
    session_id: str | None = None,
    duration_ms: float | None = None,
    exc: BaseException | str | None = None,
    details: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    **payload: Any,
) -> None:
    """Emit a structured log event with the agreed-upon schema."""

    logger = logger or LOGGER
    event: dict[str, Any] = {"step": step, "module": logger.name}
    if trace_id:
        event["trace_id"] = trace_id
    if req_id:
        event["req_id"] = req_id
    if session_id:
        event["session_id"] = session_id
    if duration_ms is not None:
        event["duration_ms"] = round(duration_ms, 3)
    if details is not None:
        event["details"] = details
    if extra:
        event.update(extra)
    event.update(payload)

    exc_info = None
    if exc is not None:
        if isinstance(exc, BaseException):
            event["exc"] = _format_exception(exc)
            exc_info = (exc.__class__, exc, exc.__traceback__)
        else:
            event["exc"] = str(exc)

    log_method = getattr(logger, level.lower(), logger.info)
    log_method(event, exc_info=exc_info)


def emit_app_startup_event() -> None:
    cwd = str(Path.cwd())
    commit = _resolve_git_commit()
    env_values = {key: os.getenv(key) for key in _ENV_KEYS_TO_LOG if os.getenv(key) is not None}
    details = {
        "env": env_values,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    payload = {
        "commit": commit,
        "user": _safe_getuid(),
        "group": _safe_getgid(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "cwd": cwd,
    }
    log_event(LOGGER, "app.startup", details=details, extra=payload)


def emit_container_context() -> None:
    runtime = os.getenv("CONTAINER_RUNTIME")
    device_requests = os.getenv("NVIDIA_VISIBLE_DEVICES")
    mounts = _read_proc_mounts()
    details = {
        "runtime": runtime,
        "device_requests": device_requests,
        "mounts": mounts,
    }
    log_event(LOGGER, "container.info", details=details)


def _read_proc_mounts(max_entries: int = 20) -> list[str]:
    mounts: list[str] = []
    try:
        content = Path("/proc/self/mounts").read_text().splitlines()
    except Exception:  # pragma: no cover - depends on permissions
        return mounts
    for line in content[:max_entries]:
        parts = line.split()
        if len(parts) >= 2:
            mounts.append(f"{parts[0]}:{parts[1]}")
    if len(content) > max_entries:
        mounts.append(f"... {len(content) - max_entries} more")
    return mounts


def emit_gpu_runtime_check() -> dict[str, Any]:
    snapshot = _capture_nvidia_smi()
    details = {
        "records": snapshot.get("records"),
        "returncode": snapshot.get("returncode"),
        "stderr": snapshot.get("stderr"),
    }
    level = "info" if snapshot.get("returncode") == 0 else "warning"
    log_event(LOGGER, "gpu.runtime_check", level=level, details=details)
    return details


def emit_gpu_torch_check() -> dict[str, Any]:
    torch_info = _collect_torch_info()
    level = "info"
    if torch_info.get("error"):
        level = "warning"
    elif torch_info.get("cuda_available") is False:
        level = "warning"
    log_event(LOGGER, "gpu.torch_check", level=level, details=torch_info)
    return torch_info


def emit_gpu_detection() -> None:
    runtime_details = emit_gpu_runtime_check()
    torch_details = emit_gpu_torch_check()
    details = {
        "runtime": runtime_details,
        "torch": torch_details,
    }
    log_event(LOGGER, "gpu.detect", details=details)


def _collect_torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_available": None,
        "version": None,
        "device_count": None,
    }
    try:
        import torch  # type: ignore import-not-found
    except Exception as error:  # pragma: no cover - optional dependency
        info["error"] = str(error)
        return info

    info["version"] = getattr(torch, "__version__", None)
    info["cuda_available"] = torch.cuda.is_available()
    info["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    info["cuda_version"] = getattr(torch.version, "cuda", None)
    return info


def emit_env_versions() -> None:
    from importlib import metadata

    packages = {
        "python": sys.version.split()[0],
        "pip": _package_version(metadata, "pip"),
        "torch": _package_version(metadata, "torch"),
        "transformers": _package_version(metadata, "transformers"),
        "sentence_transformers": _package_version(metadata, "sentence-transformers"),
        "huggingface_hub": _package_version(metadata, "huggingface-hub"),
        "chromadb": _package_version(metadata, "chromadb"),
        "numpy": _package_version(metadata, "numpy"),
    }
    log_event(LOGGER, "env.versions", details=packages)


def _package_version(metadata_module: Any, name: str) -> Optional[str]:
    try:
        return metadata_module.version(name)
    except metadata_module.PackageNotFoundError:  # type: ignore[attr-defined]
        return None
    except Exception:  # pragma: no cover - defensive guard
        return None


def emit_numpy_compat_error(module: str, error: BaseException) -> None:
    log_event(
        LOGGER,
        "compat.numpy",
        level="error",
        module=module,
        details={"module": module},
        exc=error,
        extra={"error_message": str(error)},
    )


def emit_model_path_inspection(path: str) -> None:
    path_obj = Path(path)
    exists = path_obj.exists()
    files: list[dict[str, Any]] = []
    total_size = 0
    file_count = 0
    if exists:
        for file_path in _iter_model_files(path_obj):
            try:
                stats = file_path.stat()
            except OSError as error:  # pragma: no cover - fs dependent
                log_event(
                    LOGGER,
                    "model.path.inspect",
                    level="warning",
                    details={"path": path, "error": str(error)},
                )
                continue
            size = int(stats.st_size)
            total_size += size
            file_count += 1
            files.append(
                {
                    "name": str(file_path.relative_to(path_obj)),
                    "size": size,
                    "suffix": file_path.suffix,
                    "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stats.st_mtime)),
                }
            )

    details = {
        "path": path,
        "exists": exists,
        "file_count": file_count,
        "files": files,
        "total_size_bytes": total_size,
    }
    log_event(LOGGER, "model.path.inspect", details=details)


def _iter_model_files(path: Path, max_files: int = 100) -> Iterator[Path]:
    collected = 0
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        yield file_path
        collected += 1
        if collected >= max_files:
            break


def emit_model_permissions(path: str) -> None:
    path_obj = Path(path)
    try:
        stats = path_obj.stat()
    except OSError as error:  # pragma: no cover - fs dependent
        log_event(
            LOGGER,
            "model.permissions",
            level="warning",
            details={"path": path, "error": str(error)},
        )
        return

    owner = f"{stats.st_uid}:{stats.st_gid}"
    mode = stats.st_mode
    permissions = {
        "user": bool(mode & stat.S_IRUSR),
        "group": bool(mode & stat.S_IRGRP),
        "other": bool(mode & stat.S_IROTH),
        "read_ok": os.access(path, os.R_OK),
        "write_ok": os.access(path, os.W_OK),
        "execute_ok": os.access(path, os.X_OK),
    }

    log_event(
        LOGGER,
        "model.permissions",
        details={
            "path": path,
            "owner": owner,
            "permissions": permissions,
        },
    )


def emit_model_load_start(
    *,
    provider: str,
    model_path: str,
    device_map: object,
    dtype: str | None,
    low_cpu_mem_usage: bool,
    quantization: str | None = None,
) -> None:
    gpu_snapshot = _current_gpu_snapshot()
    details = {
        "provider": provider,
        "model_path": model_path,
        "device_map": device_map,
        "dtype": dtype,
        "low_cpu_mem_use": low_cpu_mem_usage,
        "quantization": quantization,
        "gpu": gpu_snapshot,
    }
    log_event(LOGGER, "model.load.start", details=details)


def emit_model_load_progress(
    *,
    stage: str,
    attempt: int,
    total: int,
    device_map: object | None = None,
    dtype: str | None = None,
    use_gpu: bool | None = None,
    error: BaseException | str | None = None,
) -> None:
    details: dict[str, Any] = {
        "stage": stage,
        "attempt": attempt,
        "total": total,
    }
    if device_map is not None:
        details["device_map"] = device_map
    if dtype is not None:
        details["dtype"] = dtype
    if use_gpu is not None:
        details["use_gpu"] = use_gpu
    level = "error" if error else "info"
    log_event(LOGGER, "model.load.progress", level=level, details=details, exc=error)


def emit_tokenizer_event(*, path: str, duration_ms: float, error: BaseException | None = None) -> None:
    level = "error" if error else "info"
    details = {"path": path, "duration_ms": round(duration_ms, 3)}
    log_event(LOGGER, "tokenizer.load", level=level, details=details, exc=error)


def emit_model_shard_map(model_path: str) -> None:
    path_obj = Path(model_path)
    shards: list[dict[str, Any]] = []
    total = 0
    for shard in _iter_model_files(path_obj):
        size = shard.stat().st_size if shard.exists() else 0
        total += size
        target = "gpu" if shard.suffix in {".safetensors", ".bin"} else "cpu"
        shards.append({"file": str(shard.relative_to(path_obj)), "size": size, "target": target})
    details = {"path": model_path, "shards": shards, "total_size_bytes": total}
    log_event(LOGGER, "model.shard_map", details=details)


def emit_model_shard_load(
    *,
    file_name: str,
    start_ts: float,
    end_ts: float,
    error: BaseException | None = None,
) -> None:
    details = {
        "file": file_name,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_ms": round((end_ts - start_ts) * 1000.0, 3),
    }
    level = "error" if error else "debug"
    log_event(LOGGER, "model.shard.load", level=level, details=details, exc=error)


def emit_model_allocation(
    *, tensor: str, bytes_allocated: int, total_allocated_mb: Optional[float] = None
) -> None:
    details = {
        "tensor": tensor,
        "bytes": bytes_allocated,
        "total_allocated_mb": total_allocated_mb,
    }
    log_event(LOGGER, "model.alloc", level="debug", details=details)


def emit_cuda_error(error: BaseException, snapshot: dict[str, Any] | None = None) -> None:
    log_event(LOGGER, "cuda.error", level="error", details={"nvidia_smi": snapshot}, exc=error)


def emit_model_load_success(
    *,
    model_name: str,
    device_map: object,
    duration_ms: float,
    params: int | None = None,
    peak_memory_mb: float | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    details = {
        "model_name": model_name,
        "device_map": device_map,
        "params": params,
        "peak_memory_mb": peak_memory_mb,
        "config": config,
    }
    log_event(LOGGER, "model.load.success", duration_ms=duration_ms, details=details)


def emit_model_load_error(
    *,
    model_path: str,
    attempts: int,
    errors: list[dict[str, Any]],
) -> None:
    details = {
        "model_path": model_path,
        "attempts": attempts,
        "errors": errors,
    }
    log_event(LOGGER, "model.load.error", level="error", details=details)


def emit_llm_provider_init(
    *, provider: str, ready: bool, max_tokens: int | None, temperature: float | None
) -> None:
    details = {
        "provider": provider,
        "ready": ready,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    log_event(LOGGER, "llm.provider.init", details=details)


def emit_inference_request(
    *,
    req_id: str,
    session_id: str,
    prompt_preview: str,
    prompt_len: int,
    top_k: int,
    top_p: float | None,
    temperature: float,
    max_tokens: int | None,
    sources: Iterable[str],
) -> None:
    preview = prompt_preview[:120]
    details = {
        "prompt_preview": preview,
        "prompt_len": prompt_len,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "sources": list(sources),
    }
    log_event(LOGGER, "inference.request", req_id=req_id, session_id=session_id, details=details)


def emit_inference_result(
    *,
    req_id: str,
    session_id: str,
    duration_ms: float,
    model_used: str,
    answer_preview: str,
    truncated: bool,
    fallback: bool,
    tokens_generated: int | None,
) -> None:
    details = {
        "model_used": model_used,
        "answer_preview": answer_preview[:120],
        "truncated": truncated,
        "fallback": fallback,
        "tokens_generated": tokens_generated,
    }
    log_event(
        LOGGER,
        "inference.result",
        req_id=req_id,
        session_id=session_id,
        duration_ms=duration_ms,
        details=details,
    )


def emit_embeddings_event(
    *, model: str, count: int, duration_ms: float, errors: list[str] | None = None
) -> None:
    details = {
        "model": model,
        "count": count,
        "duration_ms": round(duration_ms, 3),
        "errors": errors or [],
        "per_item_ms": round(duration_ms / count, 3) if count else None,
    }
    log_event(LOGGER, "embeddings.compute", details=details)


def emit_vectorstore_event(
    step: str,
    *,
    collection: str,
    count: int,
    persist_dir: str,
    before_bytes: int | None = None,
    after_bytes: int | None = None,
    error: BaseException | None = None,
) -> None:
    details = {
        "collection": collection,
        "count": count,
        "persist_dir": persist_dir,
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
    }
    level = "error" if error else "info"
    log_event(LOGGER, step, level=level, details=details, exc=error)


def emit_retriever_event(
    *,
    query: str,
    top_k: int,
    results: list[dict[str, Any]],
    duration_ms: float,
) -> None:
    details = {
        "query_preview": query[:120],
        "top_k": top_k,
        "results": results,
    }
    log_event(LOGGER, "retriever.search", duration_ms=duration_ms, details=details)


def emit_prompt_event(
    *,
    system_prompt: str,
    sources: Iterable[str],
    context_tokens: int,
    truncated: bool,
) -> None:
    details = {
        "system_prompt_preview": system_prompt[:120],
        "sources": list(sources),
        "context_tokens": context_tokens,
        "truncated": truncated,
    }
    log_event(LOGGER, "prompt.compose", details=details)


def emit_safety_event(*, blocked: bool, rule_ids: Iterable[str], reason: str | None) -> None:
    details = {"blocked": blocked, "rule_ids": list(rule_ids), "reason": reason}
    log_event(LOGGER, "safety.filter", details=details)


def emit_ingest_event(
    step: str,
    *,
    file_name: str,
    session_id: str,
    size_bytes: int | None = None,
    duration_ms: float | None = None,
    language: str | None = None,
    pages: int | None = None,
    ocr: bool | None = None,
    chunks: int | None = None,
) -> None:
    details = {
        "file": file_name,
        "size_bytes": size_bytes,
        "duration_ms": duration_ms,
        "language": language,
        "pages": pages,
        "ocr": ocr,
        "chunks": chunks,
    }
    log_event(LOGGER, step, session_id=session_id, details=details)


def emit_exception(
    *,
    module: str,
    error: BaseException,
    req_id: str | None = None,
    session_id: str | None = None,
    suggestion: str | None = None,
) -> None:
    details = {"module": module}
    if suggestion:
        details["suggestion"] = suggestion
    log_event(
        LOGGER,
        "exception",
        level="error",
        req_id=req_id,
        session_id=session_id,
        details=details,
        exc=error,
    )


def emit_resources_snapshot() -> None:
    gpu = _current_gpu_snapshot()
    memory_info = _read_memory_info()
    cpu_load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    disk_usage = _disk_usage_summary()
    details = {
        "nvidia_smi": gpu,
        "host_mem": memory_info,
        "cpu_load_1m": cpu_load,
        "disk": disk_usage,
    }
    log_event(LOGGER, "resources.snapshot", details=details)


def _current_gpu_snapshot() -> dict[str, Any] | None:
    snapshot = _capture_nvidia_smi()
    if snapshot.get("returncode") != 0 or not snapshot.get("records"):
        return None
    return {"records": snapshot.get("records")}


def _read_memory_info() -> dict[str, Any]:
    info = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal"):
                    info["mem_total_kb"] = int(line.split()[1])
                elif line.startswith("MemAvailable"):
                    info["mem_available_kb"] = int(line.split()[1])
    except Exception:  # pragma: no cover - depends on OS
        return info
    return info


def _disk_usage_summary() -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for key in ("LLM_MODEL_PATH", "CHROMA_PERSIST_DIR"):
        value = os.getenv(key)
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.exists():
            continue
        try:
            usage = shutil.disk_usage(path)
        except Exception:  # pragma: no cover - depends on fs
            continue
        summaries[key] = {
            "path": str(path),
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
        }
    return summaries


def _collect_host_config() -> dict[str, Any]:
    container_id = os.getenv("HOSTNAME")
    if not container_id:
        return {"error": "HOSTNAME not available"}
    returncode, stdout, stderr = _run_command(["docker", "inspect", container_id])
    if returncode != 0:
        return {
            "returncode": returncode,
            "stderr": stderr or "docker inspect unavailable",
            "container_id": container_id,
        }
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        return {
            "returncode": returncode,
            "stderr": f"failed to decode docker inspect output: {error}",
            "container_id": container_id,
        }
    host_config = None
    if isinstance(payload, list) and payload:
        host_config = payload[0].get("HostConfig")  # type: ignore[index]
    return {"container_id": container_id, "host_config": host_config}


def collect_gpu_debug_snapshot() -> dict[str, Any]:
    return {
        "runtime": _capture_nvidia_smi(),
        "torch": _collect_torch_info(),
        "host_config": _collect_host_config(),
    }


@contextmanager
def traced_duration(step: str, *, logger: Optional[logging.Logger] = None, **fields: Any) -> Iterator[None]:
    start = time.perf_counter()
    log_event(logger or LOGGER, f"{step}.start", details=fields)
    try:
        yield
    except Exception as error:
        log_event(logger or LOGGER, f"{step}.error", level="error", details=fields, exc=error)
        raise
    finally:
        end = time.perf_counter()
        log_event(
            logger or LOGGER,
            f"{step}.complete",
            duration_ms=(end - start) * 1000.0,
            details=fields,
        )


def _resolve_git_commit() -> Optional[str]:
    returncode, stdout, _ = _run_command(["git", "rev-parse", "HEAD"])
    if returncode != 0:
        return None
    return stdout.strip() or None


__all__ = [
    "emit_app_startup_event",
    "emit_container_context",
    "emit_gpu_runtime_check",
    "emit_gpu_torch_check",
    "emit_gpu_detection",
    "emit_env_versions",
    "emit_numpy_compat_error",
    "emit_model_path_inspection",
    "emit_model_permissions",
    "emit_model_load_start",
    "emit_model_load_progress",
    "emit_model_shard_map",
    "emit_model_shard_load",
    "emit_model_allocation",
    "emit_model_load_success",
    "emit_model_load_error",
    "emit_tokenizer_event",
    "emit_llm_provider_init",
    "emit_inference_request",
    "emit_inference_result",
    "emit_embeddings_event",
    "emit_vectorstore_event",
    "emit_retriever_event",
    "emit_prompt_event",
    "emit_safety_event",
    "emit_ingest_event",
    "emit_exception",
    "emit_resources_snapshot",
    "collect_gpu_debug_snapshot",
    "traced_duration",
    "log_event",
]

