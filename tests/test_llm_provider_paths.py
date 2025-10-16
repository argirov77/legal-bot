"""Tests for resolving LLM model paths."""

from __future__ import annotations

from app import llm_provider


def test_normalise_model_path_resolves_relative_path(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    models_dir = repo_root / "models" / "bgpt-7b"
    models_dir.mkdir(parents=True)

    monkeypatch.setattr(llm_provider, "PROJECT_ROOT", repo_root)

    resolved = llm_provider._normalise_model_path("models/bgpt-7b")

    assert resolved == str(models_dir)


def test_normalise_model_path_resolves_docker_style_path(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    models_dir = repo_root / "models" / "bgpt-7b"
    models_dir.mkdir(parents=True)

    monkeypatch.setattr(llm_provider, "PROJECT_ROOT", repo_root)

    resolved = llm_provider._normalise_model_path("/models/bgpt-7b")

    assert resolved == str(models_dir)


def test_normalise_model_path_keeps_existing_absolute_path(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    absolute_dir = tmp_path / "weights"
    absolute_dir.mkdir()

    monkeypatch.setattr(llm_provider, "PROJECT_ROOT", repo_root)

    resolved = llm_provider._normalise_model_path(str(absolute_dir))

    assert resolved == str(absolute_dir)
