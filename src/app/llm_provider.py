"""Utilities for loading and accessing the local Large Language Model."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


try:  # pragma: no cover - optional heavy dependencies
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as import_error:  # pragma: no cover - optional heavy deps
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR: Optional[Exception] = import_error
else:  # pragma: no cover - executed when heavy deps installed
    _IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT_BG = (
    "You are a legal assistant. Answer in Bulgarian, use only provided sources. "
    "If topic not covered by sources, reply: 'Нямам достатъчно информация в заредените документи.'"
)

DEFAULT_STUB_RESPONSE = "Моделът в момента не е наличен. Моля, опитайте отново по-късно."


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


def _resolve_dtype(value: str | None) -> Optional["torch.dtype"]:
    if torch is None:
        return None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if not value:
        return torch.float16
    normalised = value.strip().lower()
    return mapping.get(normalised, torch.float16)


def _single_device_map() -> tuple[object, str]:
    if torch is None or not torch.cuda.is_available():  # pragma: no cover - depends on hw
        LOGGER.warning(
            "CUDA is not available; loading the LLM on CPU. Expect slower responses."
        )
        return {"": "cpu"}, "cpu"
    return {"": "cuda:0"}, "cuda:0"


def _auto_device_map() -> tuple[object, str]:
    if torch is None or not torch.cuda.is_available():  # pragma: no cover - depends on hw
        LOGGER.warning(
            "device_map=auto requested but GPU is not available; falling back to CPU."
        )
        return {"": "cpu"}, "cpu"
    return "auto", "cuda:auto"


def _resolve_device_map(strategy: str) -> tuple[object, str]:
    normalised = strategy.strip().lower()
    if normalised in {"single", "single:0", "single_gpu"}:
        return _single_device_map()
    if normalised == "auto":
        return _auto_device_map()
    if normalised in {"sequential", "balanced"}:
        LOGGER.warning(
            "device_map strategy '%s' is not yet supported; using single GPU fallback.",
            strategy,
        )
        return _single_device_map()
    LOGGER.warning(
        "Unknown device_map strategy '%s'; defaulting to single GPU fallback.", strategy
    )
    return _single_device_map()


def _resolve_device_strategy() -> str:
    explicit_device = os.getenv("LLM_DEVICE")
    if explicit_device:
        normalised = explicit_device.strip().lower()
        if normalised in {"cpu", "cpu:0"}:
            return "single"
        if normalised in {"cuda", "cuda:0", "gpu"}:
            return "single"
        if normalised == "auto":
            return "auto"
    map_from_env = os.getenv("LLM_DEVICE_MAP")
    if map_from_env:
        return map_from_env
    return "single"


@dataclass(slots=True)
class _LLMConfig:
    model_path: str
    device_strategy: str
    torch_dtype: Optional["torch.dtype"]
    quantization: Optional[str]


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

            LOGGER.info("loading model... (source=%s)", self._config.model_path)
            device_map, device_label = _resolve_device_map(self._config.device_strategy)

            if self._config.quantization:
                LOGGER.warning(
                    "Quantization '%s' requested but not currently configured; proceeding without it.",
                    self._config.quantization,
                )

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self._config.model_path,
                    device_map=device_map,
                    torch_dtype=self._config.torch_dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=False,
                )
                self._device_label = device_label
                self._inference_device = (
                    "cuda:0"
                    if "cuda" in device_label and torch.cuda.is_available()
                    else "cpu"
                )
            except Exception as first_error:  # pragma: no cover - depends on hw/config
                if self._config.device_strategy.strip().lower() == "auto":
                    LOGGER.warning(
                        "device_map=auto failed (%s); retrying with single GPU fallback.",
                        first_error,
                    )
                    device_map, device_label = _single_device_map()
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            self._config.model_path,
                            device_map=device_map,
                            torch_dtype=self._config.torch_dtype,
                            low_cpu_mem_usage=True,
                            trust_remote_code=False,
                        )
                        self._device_label = device_label
                        self._inference_device = (
                            "cuda:0"
                            if "cuda" in device_label and torch.cuda.is_available()
                            else "cpu"
                        )
                    except Exception as second_error:  # pragma: no cover - optional path
                        self._load_error = second_error
                        raise LLMNotReadyError(
                            "Failed to load the language model with any supported device_map."
                        ) from second_error
                else:
                    self._load_error = first_error
                    raise LLMNotReadyError("Failed to load the language model") from first_error

            try:
                tokenizer = AutoTokenizer.from_pretrained(self._config.model_path)
                if (
                    tokenizer.pad_token_id is None
                    and tokenizer.eos_token_id is not None
                ):  # pragma: no cover - configuration guard
                    tokenizer.pad_token_id = tokenizer.eos_token_id
            except Exception as error:  # pragma: no cover - depends on model files
                self._load_error = error
                raise LLMNotReadyError("Failed to load tokenizer") from error

            self._model = model
            self._tokenizer = tokenizer
            LOGGER.info("model loaded on %s", self._device_label)
            self._load_error = None

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


def _build_config() -> Optional[_LLMConfig]:
    model_path = _normalise_model_path(os.getenv("LLM_MODEL_PATH"))
    if not model_path:
        return None
    device_strategy = _resolve_device_strategy()
    torch_dtype = _resolve_dtype(os.getenv("LLM_TORCH_DTYPE", "float16"))
    quantization = os.getenv("LLM_QUANT")
    return _LLMConfig(
        model_path=model_path,
        device_strategy=device_strategy,
        torch_dtype=torch_dtype,
        quantization=quantization,
    )


def get_llm() -> LLM:
    """Return a lazily initialised LLM instance or a stub fallback."""

    global _GLOBAL_LLM

    if _GLOBAL_LLM is not None:
        return _GLOBAL_LLM

    if _env_flag("LLM_STUB"):
        LOGGER.warning("LLM_STUB flag enabled; using stub responses only.")
        _GLOBAL_STUB.update_reason("LLM_STUB flag enabled; model loading disabled.")
        _GLOBAL_LLM = _GLOBAL_STUB
        return _GLOBAL_LLM

    config = _build_config()
    if config is None:
        LOGGER.warning(
            "LLM_MODEL_PATH is not configured; falling back to stub responses."
        )
        _GLOBAL_STUB.update_reason("LLM_MODEL_PATH is not configured.")
        _GLOBAL_LLM = _GLOBAL_STUB
        return _GLOBAL_LLM

    _GLOBAL_LLM = TransformersLLM(config, stub=_GLOBAL_STUB)
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

    model_path = os.getenv("LLM_MODEL_PATH")
    if not model_path:
        LOGGER.info(
            "Startup model loading requested but LLM_MODEL_PATH is not configured; skipping."
        )
        return False, None

    llm = get_llm()
    if isinstance(llm, LLMStub):
        LOGGER.info("Startup model loading requested but LLM stub backend is active; skipping.")
        return False, None

    LOGGER.info("attempting to load model %s", llm.model_name)
    try:
        llm.preload()
    except Exception as error:  # pragma: no cover - depends on environment
        LOGGER.exception("failed to load model %s", llm.model_name)
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

