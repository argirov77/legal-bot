"""Utilities for constructing prompts for the Bulgarian legal assistant."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

_BASE_DIR = Path(__file__).resolve().parents[2]
_SYSTEM_PROMPT_PATH = _BASE_DIR / "prompts" / "bg" / "system.txt"
_USER_PROMPT_PATH = _BASE_DIR / "prompts" / "bg" / "user.md"


def _load_template(path: Path) -> str:
    """Read and trim the contents of a template file."""
    return path.read_text(encoding="utf-8").strip()


_SYSTEM_TEXT = _load_template(_SYSTEM_PROMPT_PATH)
_USER_TEMPLATE = _load_template(_USER_PROMPT_PATH)


def build_prompt(question: str, contexts: List[Dict]) -> str:
    """Compose the full prompt used for answering a user's legal question."""

    if question is None:
        raise ValueError("question must not be None")

    sorted_contexts = sorted(contexts or [], key=lambda ctx: ctx.get("score", 0), reverse=True)

    context_sections: List[str] = []
    for index, context in enumerate(sorted_contexts, start=1):
        content = str(context.get("content", "")).strip()
        if not content:
            continue

        metadata = context.get("metadata") or {}
        citation = metadata.get("source") or context.get("id") or f"context-{index}"
        section = f"[citation:{citation}] {content}"
        context_sections.append(section)

    if context_sections:
        contexts_block = "\n\n".join(context_sections)
    else:
        contexts_block = "Няма наличен контекст."

    user_block = _USER_TEMPLATE.format(question=question.strip())

    return f"{_SYSTEM_TEXT}\n\n{contexts_block}\n\n{user_block}".strip()


__all__ = ["build_prompt"]
