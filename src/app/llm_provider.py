"""Utilities for loading and accessing the local Large Language Model."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from app.telemetry import (
    emit_gpu_runtime_check,
    emit_gpu_torch_check,
    emit_cuda_error,
    emit_exception,
    emit_llm_provider_init,
    emit_model_load_start,
    emit_model_load_progress,
    emit_model_load_error,
    emit_model_load_success,
    emit_model_path_inspection,
    emit_model_permissions,
    emit_model_shard_map,
    emit_numpy_compat_error,
    emit_tokenizer_event,
)


try:  # pragma: no cover - optional heavy dependencies
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as import_error:  # pragma: no cover - optional heavy deps
    emit_numpy_compat_error("torch/transformers", import_error)
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR: Optional[Exception] = import_error
else:  # pragma: no cover - executed when heavy deps installed
    _IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH_ENV_KEYS: tuple[str, str, str] = (
    "LLM_MODEL_PATH",
    "LLM_BG1_PATH",
    "LLM_BG2_PATH",
)

_INSPECTED_PATHS: set[str] = set()
_GPU_DIAGNOSTICS_EMITTED = False

SYSTEM_PROMPT_BG = (
    "You are a legal assistant. Answer in Bulgarian, use only provided sources. "
    "If topic not covered by sources, reply: 'Нямам достатъчно информация в заредените документи.'"
)

DEFAULT_STUB_RESPONSE = "Моделът в момента не е наличен. Моля, опитайте по-късно."


@dataclass(slots=True)
class LLMStatus:
    """Structured status information about the configured LLM backend."""

    model_loaded: bool
    model_name: str
    device: str
    error: Optional[str] = None


class LLMError(RuntimeError):
    """Base exception raised for LLM provider issues."""


class LLMNotReadyError(LLMError):
    """Raised when the model cannot be loaded or is unavailable."""


class LLMGenerationError(LLMError):
    """Raised when text generation fails unexpectedly."""


class LLM:
    """Common interface exposed by language model implementations."""

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate a response for the provided prompt."""

        raise NotImplementedError

    @property
    def model_loaded(self) -> bool:
        """Return ``True`` when the underlying model weights are loaded."""

        return False

    @property
    def model_name(self) -> str:
        """Human-readable identifier describing the model."""

        return "stub"

    @property
    def device(self) -> str:
        """Device where the model resides (``cpu`` / ``cuda:0`` / etc.)."""

        return "cpu"

    @property
    def last_error(self) -> Optional[str]:
        """Return the most recent loading error, if any."""

        return None

    def preload(self) -> None:
        """Eagerly load the model weights when supported."""

        return None

    def status(self) -> LLMStatus:
        """Return structured diagnostic information for health checks."""

        return LLMStatus(
            model_loaded=self.model_loaded,
            model_name=self.model_name,
            device=self.device,
            error=self.last_error,
        )


class LLMStub(LLM):
    """Fallback implementation returning a friendly Bulgarian message."""

    def __init__(
        self,
        message: str = DEFAULT_STUB_RESPONSE,
        *,
        reason: str | None = None,
    ) -> None:
        self._message = message
        self._reason = reason or "LLM stub is active (model not configured)."

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        return self._message

    @property
    def last_error(self) -> Optional[str]:
        return self._reason

    def update_reason(self, reason: str) -> None:
        self._reason = reason


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_float(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _format_device_map(device_map: object) -> str:
    if isinstance(device_map, str):
        return device_map
    if isinstance(device_map, dict):
        return ", ".join(f"{key or '<model>'}->{value}" for key, value in device_map.items())
    return repr(device_map)


def _resolve_llm_device() -> str:
    want = os.getenv("LLM_DEVICE", "auto").strip().lower()
    if want in {"cuda", "gpu"}:
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    if want == "cpu":
        return "cpu"
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def _log_memory_snapshot(device_label: str) -> None:
    if torch is None:
        return

    if device_label.startswith("cuda") and torch.cuda.is_available():  # pragma: no cover
        try:
            device_index = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device_index).total_memory
            allocated = torch.cuda.memory_allocated(device_index)
            reserved = torch.cuda.memory_reserved(device_index)
        except Exception as error:  # pragma: no cover - depends on drivers
            LOGGER.debug("Unable to collect CUDA memory statistics: %s", error)
            return

        gib = float(1024**3)
        LOGGER.info(
            "CUDA memory stats: allocated=%.2f GiB reserved=%.2f GiB total=%.2f GiB",
            allocated / gib,
            reserved / gib,
            total / gib,
        )


@dataclass(slots=True)
class _LLMConfig:
    model_path: str


class TransformersLLM(LLM):
    """Lazy-loading wrapper around ``AutoModelForCausalLM``."""

    def __init__(self, config: _LLMConfig, *, stub: Optional[LLM] = None) -> None:
        self._config = config
        self._model: Optional["AutoModelForCausalLM"] = None
        self._tokenizer: Optional["AutoTokenizer"] = None
        self._lock = threading.RLock()
        self._load_error: Optional[Exception] = None
        self._device_label = "cpu"
        self._inference_device = "cpu"
        self._stub = stub or LLMStub()

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        if self._model is not None:
            return getattr(self._model.config, "_name_or_path", self._config.model_path)
        return self._config.model_path

    @property
    def device(self) -> str:
        return self._device_label

    @property
    def last_error(self) -> Optional[str]:
        if self._load_error is None:
            return None
        return str(self._load_error)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
                base_error = _IMPORT_ERROR or RuntimeError(
                    "PyTorch/Transformers are not available in the current environment"
                )
                self._load_error = base_error
                raise LLMNotReadyError(
                    "PyTorch/Transformers are not available in the current environment"
                ) from base_error

            global _GPU_DIAGNOSTICS_EMITTED
            if not _GPU_DIAGNOSTICS_EMITTED:
                emit_gpu_runtime_check()
                emit_gpu_torch_check()
                _GPU_DIAGNOSTICS_EMITTED = True

            device = _resolve_llm_device()
            use_cuda = device == "cuda"
            device_map: object = "auto" if use_cuda else "cpu"
            torch_dtype: object = "auto" if use_cuda else torch.float32
            dtype_label = str(torch_dtype)
            total_attempts = 1
            attempt_index = 1

            emit_model_load_progress(
                stage="init",
                attempt=0,
                total=1,
                device_map=_format_device_map(device_map),
                dtype=dtype_label,
                use_gpu=use_cuda,
            )

            load_started = time.perf_counter()
            emit_model_shard_map(self._config.model_path)

            emit_model_load_progress(
                stage="attempt_start",
                attempt=attempt_index,
                total=total_attempts,
                device_map=_format_device_map(device_map),
                dtype=dtype_label,
                use_gpu=use_cuda,
            )
            LOGGER.info(
                "trying to load LLM from %s (device_map=%s, dtype=%s, low_cpu_mem_usage=True)",
                self._config.model_path,
                _format_device_map(device_map),
                dtype_label,
            )
            LOGGER.info(
                "llm.load.args",
                extra={
                    "details": {
                        "device": device,
                        "device_map": device_map,
                        "torch_dtype": str(torch_dtype),
                    }
                },
            )
            emit_model_load_start(
                provider="transformers",
                model_path=self._config.model_path,
                device_map=_format_device_map(device_map),
                dtype=dtype_label,
                low_cpu_mem_usage=True,
                quantization=None,
            )

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self._config.model_path,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=False,
                )
            except Exception as error:  # pragma: no cover - depends on hw/config
                LOGGER.warning("Failed to load LLM on %s: %s", device, error)
                if "CUDA out of memory" in str(error):
                    emit_cuda_error(error, snapshot=None)
                else:
                    emit_exception(module=__name__, error=error)
                emit_model_load_progress(
                    stage="attempt_error",
                    attempt=attempt_index,
                    total=total_attempts,
                    device_map=_format_device_map(device_map),
                    dtype=dtype_label,
                    use_gpu=use_cuda,
                    error=error,
                )
                emit_model_load_error(
                    model_path=self._config.model_path,
                    attempts=1,
                    errors=[
                        {
                            "device": device,
                            "error": str(error),
                        }
                    ],
                )
                self._load_error = error
                self._stub.update_reason(f"Failed to load model on {device}: {error}")
                raise LLMNotReadyError("Failed to load the language model") from error

            try:
                LOGGER.info(
                    "llm.device.map",
                    extra={"details": getattr(model, "hf_device_map", None)},
                )
            except Exception:  # pragma: no cover - defensive guard
                pass

            self._device_label = device
            self._inference_device = (
                "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"
            )
            emit_model_load_progress(
                stage="attempt_success",
                attempt=attempt_index,
                total=total_attempts,
                device_map=_format_device_map(device_map),
                dtype=dtype_label,
                use_gpu=use_cuda,
            )
            successful_map: object | None = device_map

            tokenizer_started = time.perf_counter()
            try:
                tokenizer = AutoTokenizer.from_pretrained(self._config.model_path)
                if (
                    tokenizer.pad_token_id is None
                    and tokenizer.eos_token_id is not None
                ):  # pragma: no cover - configuration guard
                    tokenizer.pad_token_id = tokenizer.eos_token_id
            except Exception as error:  # pragma: no cover - depends on model files
                emit_tokenizer_event(
                    path=self._config.model_path,
                    duration_ms=(time.perf_counter() - tokenizer_started) * 1000.0,
                    error=error,
                )
                self._load_error = error
                self._stub.update_reason(f"Failed to load tokenizer: {error}")
                raise LLMNotReadyError("Failed to load tokenizer") from error

            emit_tokenizer_event(
                path=self._config.model_path,
                duration_ms=(time.perf_counter() - tokenizer_started) * 1000.0,
                error=None,
            )

            self._model = model
            self._tokenizer = tokenizer
            LOGGER.info("model loaded on %s", self._device_label)
            _log_memory_snapshot(self._device_label)
            self._load_error = None

            duration_ms = (time.perf_counter() - load_started) * 1000.0
            params = None
            if hasattr(self._model, "num_parameters"):
                try:
                    params = int(self._model.num_parameters())  # type: ignore[call-arg]
                except Exception:  # pragma: no cover - depends on backend
                    params = None
            config_payload = None
            if getattr(self._model, "config", None) is not None:
                try:
                    config_payload = {
                        "model_type": getattr(self._model.config, "model_type", None),
                        "torch_dtype": str(getattr(self._model.config, "torch_dtype", None)),
                        "max_position_embeddings": getattr(
                            self._model.config, "max_position_embeddings", None
                        ),
                    }
                except Exception:  # pragma: no cover - defensive guard
                    config_payload = None

            emit_model_load_success(
                model_name=self.model_name,
                device_map=_format_device_map(successful_map or {}),
                duration_ms=duration_ms,
                params=params,
                peak_memory_mb=None,
                config=config_payload,
            )
            emit_model_load_progress(
                stage="complete",
                attempt=attempt_index,
                total=total_attempts,
                device_map=_format_device_map(successful_map or {}),
                dtype=dtype_label,
                use_gpu=use_cuda,
            )

    def _build_prompt(self, user_prompt: str) -> str:
        user_content = user_prompt.strip()
        if not user_content:
            user_content = "(няма въпрос)"
        return f"{SYSTEM_PROMPT_BG.strip()}\n\n{user_content}"

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            self._ensure_loaded()
        except LLMNotReadyError as error:  # pragma: no cover - depends on env
            LOGGER.warning("falling back to stub: %s", error)
            self._stub.update_reason(str(error))
            return self._stub.generate(prompt, max_tokens, temperature)

        if self._model is None or self._tokenizer is None:
            LOGGER.warning("LLM not fully initialised; returning stub response")
            return self._stub.generate(prompt, max_tokens, temperature)

        effective_max_tokens = max_tokens if max_tokens and max_tokens > 0 else 256
        do_sample = temperature > 0.0

        try:
            tokenizer_inputs = self._tokenizer(
                self._build_prompt(prompt),
                return_tensors="pt",
                truncation=True,
                max_length=getattr(self._tokenizer, "model_max_length", 4096),
            )
            if torch is not None:
                tokenizer_inputs = tokenizer_inputs.to(self._inference_device)

            output_ids = self._model.generate(
                **tokenizer_inputs,
                max_new_tokens=effective_max_tokens,
                temperature=max(0.0, float(temperature)),
                do_sample=do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        except Exception as error:  # pragma: no cover - depends on runtime behaviour
            LOGGER.exception("LLM generation failed")
            raise LLMGenerationError("LLM generation failed") from error

        input_length = tokenizer_inputs["input_ids"].shape[1]
        generated_sequence = output_ids[0, input_length:]
        try:
            text = self._tokenizer.decode(generated_sequence, skip_special_tokens=True)
        except Exception as error:  # pragma: no cover - defensive guard
            LOGGER.exception("Failed to decode model output")
            raise LLMGenerationError("Failed to decode model output") from error

        return text.strip()

    def preload(self) -> None:
        self._ensure_loaded()


_GLOBAL_LLM: Optional[LLM] = None
_GLOBAL_STUB = LLMStub()


def _normalise_model_path(raw_path: str | None) -> Optional[str]:
    if raw_path is None:
        return None

    candidate = raw_path.strip()
    if not candidate:
        return None

    expanded = os.path.expanduser(candidate)
    path = Path(expanded)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if path.exists():
        return str(path)

    docker_root = Path("/models")
    fallback_path: Optional[Path] = None

    try:
        relative_to_docker = Path(expanded).relative_to(docker_root)
    except ValueError:
        relative_to_docker = None

    if relative_to_docker is not None:
        fallback_candidate = PROJECT_ROOT / "models" / relative_to_docker
        if fallback_candidate.exists():
            fallback_path = fallback_candidate
    elif expanded.startswith(str(docker_root)):
        relative_suffix = expanded[len(str(docker_root)) :].lstrip(os.sep)
        if relative_suffix:
            fallback_candidate = PROJECT_ROOT / "models" / relative_suffix
            if fallback_candidate.exists():
                fallback_path = fallback_candidate

    if fallback_path is not None:
        LOGGER.info(
            "Resolved LLM_MODEL_PATH '%s' to repository directory '%s'.",
            raw_path,
            fallback_path,
        )
        return str(fallback_path)

    LOGGER.warning(
        "Configured LLM_MODEL_PATH '%s' does not exist (resolved to '%s').",
        raw_path,
        path,
    )
    return str(path)


def _resolve_model_path_from_env() -> Optional[str]:
    for key in MODEL_PATH_ENV_KEYS:
        raw_value = os.getenv(key)
        if raw_value is None:
            continue

        candidate = _normalise_model_path(raw_value)
        if not candidate:
            continue

        if key != "LLM_MODEL_PATH":
            LOGGER.info("Resolved model path via %s: %s", key, candidate)

        if candidate not in _INSPECTED_PATHS:
            _INSPECTED_PATHS.add(candidate)
            emit_model_path_inspection(candidate)
            emit_model_permissions(candidate)

        return candidate

    return None


def _build_config() -> Optional[_LLMConfig]:
    model_path = _resolve_model_path_from_env()
    if not model_path:
        return None
    return _LLMConfig(model_path=model_path)


def get_llm() -> LLM:
    """Return a lazily initialised LLM instance or a stub fallback."""

    global _GLOBAL_LLM

    if _GLOBAL_LLM is not None:
        return _GLOBAL_LLM

    if _env_flag("LLM_STUB"):
        LOGGER.warning("LLM_STUB flag enabled; using stub responses only.")
        _GLOBAL_STUB.update_reason("LLM_STUB flag enabled; model loading disabled.")
        emit_llm_provider_init(
            provider="stub",
            ready=False,
            max_tokens=_env_int("LLM_MAX_TOKENS"),
            temperature=_env_float("LLM_TEMPERATURE"),
        )
        _GLOBAL_LLM = _GLOBAL_STUB
        return _GLOBAL_LLM

    config = _build_config()
    if config is None:
        LOGGER.warning(
            "Model path variables %s are not configured; using stub responses.",
            ", ".join(MODEL_PATH_ENV_KEYS),
        )
        _GLOBAL_STUB.update_reason(
            "LLM_MODEL_PATH/LLM_BG1_PATH/LLM_BG2_PATH are not configured."
        )
        emit_llm_provider_init(
            provider="stub",
            ready=False,
            max_tokens=_env_int("LLM_MAX_TOKENS"),
            temperature=_env_float("LLM_TEMPERATURE"),
        )
        _GLOBAL_LLM = _GLOBAL_STUB
        return _GLOBAL_LLM

    _GLOBAL_LLM = TransformersLLM(config, stub=_GLOBAL_STUB)
    emit_llm_provider_init(
        provider="transformers",
        ready=False,
        max_tokens=_env_int("LLM_MAX_TOKENS"),
        temperature=_env_float("LLM_TEMPERATURE"),
    )
    return _GLOBAL_LLM


def _should_attempt_startup_load() -> bool:
    return _env_flag("INSTALL_HEAVY") or _env_flag("FORCE_LOAD_ON_START")


def load_llm_on_startup() -> Tuple[bool, Optional[Exception]]:
    """Attempt to load the configured LLM eagerly when requested.

    Returns a tuple ``(attempted, error)`` describing whether an attempt was made and
    the exception that occurred (if any).
    """

    if not _should_attempt_startup_load():
        return False, None

    model_path = _resolve_model_path_from_env()
    if not model_path:
        LOGGER.info(
            "Startup model loading requested but no model path variables (%s) are configured; skipping.",
            ", ".join(MODEL_PATH_ENV_KEYS),
        )
        return False, None

    llm = get_llm()
    if isinstance(llm, LLMStub):
        LOGGER.info("Startup model loading requested but LLM stub backend is active; skipping.")
        return False, None

    LOGGER.info("trying to load LLM from %s", model_path)
    try:
        llm.preload()
    except Exception as error:  # pragma: no cover - depends on environment
        LOGGER.exception("failed to load model %s", llm.model_name)
        emit_exception(module=__name__, error=error)
        return True, error

    LOGGER.info("model loaded")
    return True, None


def get_llm_status() -> LLMStatus:
    """Return structured status information about the configured LLM."""

    llm = get_llm()
    return llm.status()


__all__ = [
    "DEFAULT_STUB_RESPONSE",
    "LLM",
    "LLMStatus",
    "LLMError",
    "LLMGenerationError",
    "LLMNotReadyError",
    "LLMStub",
    "SYSTEM_PROMPT_BG",
    "get_llm_status",
    "get_llm",
    "load_llm_on_startup",
]

