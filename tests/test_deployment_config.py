"""Deployment verification scripts use configuration without contacting AWS."""

from pathlib import Path
import runpy
from unittest.mock import MagicMock

import mlflow
import pytest

from src.db import config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["test_mlflow.py", "test_aws_v3_artefact.py"])
def script(request, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    calls = {}
    for name in (
        "set_tracking_uri", "set_experiment", "start_run",
        "log_param", "log_metric", "log_artifact",
    ):
        calls[name] = MagicMock()
        monkeypatch.setattr(mlflow, name, calls[name])
    return ROOT / "deploy_mlflow_aws" / request.param, calls


def test_import_has_no_side_effects(script, monkeypatch):
    path, calls = script
    loader = MagicMock(side_effect=AssertionError("Configuration loaded on import"))
    monkeypatch.setattr(config, "load_dotenv", loader)
    runpy.run_path(str(path), run_name="import_check")
    loader.assert_not_called()
    for call in calls.values():
        call.assert_not_called()


@pytest.mark.parametrize("value", [None, ""])
def test_missing_uri_fails_before_mlflow_calls(script, monkeypatch, value):
    path, calls = script
    if value is not None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", value)
    with pytest.raises(RuntimeError, match=r"Set MLFLOW_TRACKING_URI.*environment or .env"):
        runpy.run_path(str(path), run_name="__main__")
    for call in calls.values():
        call.assert_not_called()


@pytest.mark.parametrize("source", ["environment", "dotenv"])
def test_configured_entry_point_preserves_verification(script, monkeypatch, tmp_path, source):
    path, calls = script
    dotenv_uri = "https://dotenv.example.invalid"
    (tmp_path / ".env").write_text(f"MLFLOW_TRACKING_URI={dotenv_uri}\n", encoding="utf-8")
    expected_uri = dotenv_uri
    if source == "environment":
        expected_uri = "https://environment.example.invalid"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", expected_uri)

    uploaded = []

    def inspect_artifact(filename):
        artifact = Path(filename)
        assert artifact.name == "test_artifact_5mb.bin"
        assert artifact.stat().st_size == 5 * 1024 * 1024
        uploaded.append(artifact)

    calls["log_artifact"].side_effect = inspect_artifact
    runpy.run_path(str(path), run_name="__main__")

    calls["set_tracking_uri"].assert_called_once_with(expected_uri)
    if path.name == "test_mlflow.py":
        calls["set_experiment"].assert_called_once_with("Sprint 3 Test")
        calls["start_run"].assert_called_once_with(run_name="PBI-06 Deployment Verification")
        calls["log_param"].assert_called_once_with("model", "test_model")
        calls["log_metric"].assert_called_once_with("accuracy", 0.95)
        calls["log_artifact"].assert_not_called()
    else:
        calls["set_experiment"].assert_called_once_with("100mb limit test - 5mb")
        calls["start_run"].assert_called_once_with(run_name="PBI-06 Artifact Upload Verification")
        calls["log_param"].assert_called_once_with("artifact_size_mb", 5)
        calls["log_metric"].assert_not_called()
        calls["log_artifact"].assert_called_once()
        assert len(uploaded) == 1
        assert not uploaded[0].exists()  # Temporary artifact is cleaned up.
