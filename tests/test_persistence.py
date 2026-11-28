"""Offline persistence checks: all MLflow server and artifact calls are mocked."""

import json
from pathlib import Path
import runpy
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import numpy as np
import pytest

from deploy_mlflow_aws import persistence_setup as setup
from deploy_mlflow_aws import persistence_verify as verify


class Page(list):
    def __init__(self, items=(), token=None):
        super().__init__(items)
        self.token = token


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(setup, "load_dotenv", MagicMock())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://tracking.example.invalid")
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.setattr(setup.mlflow, "active_run", MagicMock(return_value=None))
    client = MagicMock()
    monkeypatch.setattr(setup.mlflow, "MlflowClient", MagicMock(return_value=client))
    for name in ["set_tracking_uri", "set_registry_uri", "log_param", "log_metric", "log_text"]:
        monkeypatch.setattr(setup.mlflow, name, MagicMock())
    run = NS(info=NS(run_id="test-run"))
    start = MagicMock()
    start.return_value.__enter__.return_value = run
    monkeypatch.setattr(setup.mlflow, "start_run", start)
    monkeypatch.setattr(setup.mlflow.sklearn, "log_model", MagicMock(
        return_value=NS(model_uri="models:/m-test")
    ))
    monkeypatch.setattr(setup.mlflow, "register_model", MagicMock(return_value=NS(version="7")))
    monkeypatch.setattr(setup.mlflow.sklearn, "load_model", MagicMock())
    client.get_experiment_by_name.return_value = None
    client.search_registered_models.return_value = Page()
    client.search_experiments.return_value = Page([NS(experiment_id="old-experiment")])
    client.search_runs.return_value = Page([NS(info=NS(run_id="old-run"))])
    client.create_experiment.return_value = "new-experiment"
    return client, tmp_path / "persistence_reference.json"


def test_setup_saves_evidence_and_preserves_existing_objects(offline):
    client, reference = offline
    evidence = setup.setup(reference)
    assert json.loads(reference.read_text()) == evidence
    assert evidence["original_run_ids"] == ["old-run"]
    assert evidence["run_id"] == "test-run"
    assert evidence["model_version"] == "7"
    assert evidence["expected_predictions"] == [0, 1]
    assert "tracking_uri" not in evidence
    client.create_experiment.assert_called_once_with(setup.EXPERIMENT)
    client.create_registered_model.assert_called_once_with(setup.MODEL)
    setup.mlflow.start_run.assert_called_once_with(
        experiment_id="new-experiment", run_name="persistence-check"
    )
    setup.mlflow.log_param.assert_called_once_with("purpose", "persistence")
    setup.mlflow.log_metric.assert_called_once_with("accuracy", 1.0)
    setup.mlflow.log_text.assert_called_once_with(evidence["artifact_text"], evidence["artifact_path"])
    setup.mlflow.register_model.assert_called_once_with("models:/m-test", setup.MODEL)
    logged = setup.mlflow.sklearn.log_model.call_args
    assert logged.kwargs["name"] == "model"
    assert logged.kwargs["skops_trusted_types"] == ["sklearn.tree._tree.Tree"]
    assert logged.args[0].random_state == 42
    np.testing.assert_array_equal(logged.args[0].predict([[0.5], [2.5]]), [0, 1])
    client.set_registered_model_alias.assert_not_called()
    client.update_registered_model.assert_not_called()
    client.update_experiment.assert_not_called()


@pytest.mark.parametrize("existing", ["reference", "experiment", "model"])
def test_setup_refuses_to_overwrite_existing_objects(offline, existing):
    client, reference = offline
    if existing == "reference":
        reference.write_text("keep me")
    elif existing == "experiment":
        client.get_experiment_by_name.return_value = NS(experiment_id="existing")
    else:
        client.search_registered_models.return_value = Page([NS(name=setup.MODEL)])
    with pytest.raises(RuntimeError, match="already exists"):
        setup.setup(reference)
    client.create_experiment.assert_not_called()
    client.create_registered_model.assert_not_called()
    setup.mlflow.start_run.assert_not_called()
    if existing == "reference":
        assert reference.read_text() == "keep me"


def test_existing_runs_follows_experiment_and_run_pages(offline):
    client, _ = offline
    client.search_experiments.side_effect = [
        Page([NS(experiment_id="a")], "next-experiment"), Page([NS(experiment_id="b")]),
    ]
    client.search_runs.side_effect = [
        Page([NS(info=NS(run_id="1"))], "next-run"),
        Page([NS(info=NS(run_id="2"))]), Page([NS(info=NS(run_id="3"))]),
    ]
    assert setup.existing_run_ids(client) == ["1", "2", "3"]
    assert client.search_experiments.call_args_list[1].kwargs["page_token"] == "next-experiment"
    assert client.search_runs.call_args_list[1].kwargs["page_token"] == "next-run"
    assert client.search_runs.call_args.kwargs["run_view_type"] == setup.ViewType.ALL


@pytest.mark.parametrize("source", ["environment", "active_run"])
def test_setup_does_not_resume_existing_run(offline, monkeypatch, source):
    _, reference = offline
    if source == "environment":
        monkeypatch.setenv("MLFLOW_RUN_ID", "existing-run")
    else:
        setup.mlflow.active_run.return_value = NS(info=NS(run_id="existing-run"))
    with pytest.raises(RuntimeError, match="active MLflow run"):
        setup.setup(reference)
    setup.mlflow.MlflowClient.assert_not_called()
    setup.mlflow.start_run.assert_not_called()


@pytest.fixture
def evidence(offline):
    client, reference = offline
    data = setup.setup(reference)
    run = NS(info=NS(run_id="test-run", experiment_id="new-experiment", status="FINISHED"),
             data=NS(params={"purpose": "persistence"}, metrics={"accuracy": 1.0}))
    client.get_run.side_effect = lambda run_id: NS(info=NS(run_id=run_id)) if run_id == "old-run" else run
    client.get_model_version.return_value = NS(version="7", run_id="test-run")

    def download(run_id, artifact_path, directory):
        assert run_id == "test-run"
        assert artifact_path == "persistence.txt"
        target = Path(directory) / artifact_path
        target.write_text(data["artifact_text"], encoding="utf-8")
        return str(target)

    client.download_artifacts.side_effect = download
    setup.mlflow.sklearn.load_model.return_value.predict.return_value = np.array([0, 1])
    return client, reference, run


def test_verify_checks_exact_version_without_writes(evidence, capsys):
    client, reference, _ = evidence
    client.reset_mock()
    setup.mlflow.register_model.reset_mock()
    assert verify.main(["--reference", str(reference)]) == 0
    output = capsys.readouterr().out
    assert output.count("PASS:") == 7
    assert "FAIL:" not in output
    client.get_model_version.assert_called_once_with(setup.MODEL, "7")
    setup.mlflow.sklearn.load_model.assert_called_once_with(f"models:/{setup.MODEL}/7")
    client.create_experiment.assert_not_called()
    client.create_registered_model.assert_not_called()
    client.set_registered_model_alias.assert_not_called()
    setup.mlflow.register_model.assert_not_called()


@pytest.mark.parametrize("failure,label", [
    ("original", "original run old-run"),
    ("parameter", "test run parameters"),
    ("metric", "test run metrics"),
    ("artifact", "text artifact contents"),
    ("version", "exact registered model version"),
    ("prediction", "registered model predictions"),
])
def test_verify_reports_failures_and_continues(evidence, capsys, failure, label):
    client, reference, run = evidence
    if failure == "original":
        client.get_run.side_effect = lambda run_id: (_ for _ in ()).throw(
            RuntimeError("unavailable")
        ) if run_id == "old-run" else run
    elif failure == "parameter":
        run.data.params["purpose"] = "changed"
    elif failure == "metric":
        run.data.metrics["accuracy"] = 0.5
    elif failure == "artifact":
        original = client.download_artifacts.side_effect

        def corrupt(*args):
            path = original(*args)
            Path(path).write_text("changed")
            return path

        client.download_artifacts.side_effect = corrupt
    elif failure == "version":
        client.get_model_version.return_value.run_id = "different-run"
    else:
        setup.mlflow.sklearn.load_model.return_value.predict.return_value = np.array([1, 0])
    assert verify.main(["--reference", str(reference)]) == 1
    output = capsys.readouterr().out
    assert f"FAIL: {label}" in output
    assert output.count("PASS:") + output.count("FAIL:") == 7


def test_verify_missing_reference_does_not_contact_server(offline, capsys):
    _, reference = offline
    assert verify.main(["--reference", str(reference)]) == 1
    assert "FAIL: verification prerequisites" in capsys.readouterr().out
    setup.mlflow.MlflowClient.assert_not_called()


def test_missing_configuration_fails_without_contacting_server(offline, monkeypatch, capsys):
    _, reference = offline
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    assert setup.main(["--reference", str(reference)]) == 1
    assert "MLFLOW_TRACKING_URI" in capsys.readouterr().out
    setup.mlflow.MlflowClient.assert_not_called()


def test_tracking_and_registry_use_same_configured_server(offline):
    setup.configure_tracking()
    uri = "https://tracking.example.invalid"
    setup.load_dotenv.assert_called_once()
    setup.mlflow.set_tracking_uri.assert_called_once_with(uri)
    setup.mlflow.set_registry_uri.assert_called_once_with(uri)
    setup.mlflow.MlflowClient.assert_called_once_with(tracking_uri=uri, registry_uri=uri)


@pytest.mark.parametrize("filename", ["persistence_setup.py", "persistence_verify.py"])
def test_import_does_not_load_configuration_or_contact_server(offline, filename):
    runpy.run_path(str(setup.REFERENCE.with_name(filename)), run_name="import_check")
    setup.load_dotenv.assert_not_called()
    setup.mlflow.MlflowClient.assert_not_called()


def test_model_logging_failure_reports_stage_and_mlflow_error(offline, capsys):
    from mlflow.exceptions import MlflowException

    client, reference = offline
    setup.mlflow.sklearn.log_model.side_effect = MlflowException.invalid_parameter_value(
        "The saved sklearn model references untrusted types: sklearn.tree._tree.Tree",
    )
    assert setup.main(["--reference", str(reference)]) == 1
    output = capsys.readouterr().out
    assert "Setup failed during: serialize and upload decision tree model" in output
    assert "MlflowException" in output
    assert "INVALID_PARAMETER_VALUE" in output
    assert "sklearn.tree._tree.Tree" in output
    assert "Run ID: test-run" in output
    assert "do not rerun setup blindly" in output
    assert not reference.exists()
    client.create_registered_model.assert_not_called()
    setup.mlflow.register_model.assert_not_called()
    assert setup.mlflow.start_run.return_value.__exit__.call_args.args[0] is setup.SetupStepError


def test_registry_failure_is_distinguished_from_model_logging_failure(offline, capsys):
    from mlflow.exceptions import MlflowException

    _, reference = offline
    setup.mlflow.register_model.side_effect = MlflowException("Registry unavailable")
    assert setup.main(["--reference", str(reference)]) == 1
    output = capsys.readouterr().out
    assert "Setup failed during: register model version" in output
    assert "Registry unavailable" in output
    assert setup.mlflow.start_run.return_value.__exit__.call_args.args == (None, None, None)
    assert not reference.exists()


def test_error_diagnostics_redact_credentials_and_preserve_causes(offline, monkeypatch, capsys):
    from urllib.parse import quote, quote_plus
    from mlflow.exceptions import MlflowException

    _, reference = offline
    secret = "synthetic p@ss/+word"
    monkeypatch.setenv("DB_PASSWORD", secret)
    monkeypatch.setenv("AWS_SESSION_TOKEN", "synthetic-session-token")
    cause = ValueError("password='other-secret'; Authorization: Bearer hidden-bearer")
    error = MlflowException(
        f"untrusted Tree; {secret}; {quote(secret, safe='')}; {quote_plus(secret, safe='')}; "
        "https://user:hidden-password@example.invalid/path?X-Amz-Signature=hidden-signature; "
        "synthetic-session-token; [parameters: {'password': 'server-secret'}]"
    )
    error.__cause__ = cause
    setup.mlflow.sklearn.log_model.side_effect = error
    assert setup.main(["--reference", str(reference)]) == 1
    output = capsys.readouterr().out
    for value in [secret, quote(secret, safe=""), quote_plus(secret, safe=""),
                  "other-secret", "hidden-bearer", "hidden-password", "hidden-signature",
                  "synthetic-session-token", "server-secret"]:
        assert value not in output
    assert "untrusted Tree" in output
    assert "ValueError" in output
    assert "REDACTED" in output


def test_recovery_reuses_only_test_experiment_and_preserves_failed_run(offline):
    client, reference = offline
    client.get_experiment_by_name.return_value = NS(
        experiment_id="partial-experiment", lifecycle_stage="active"
    )
    client.search_runs.return_value = Page([
        NS(info=NS(run_id="old-run")), NS(info=NS(run_id="failed-run", status="FAILED")),
    ])
    assert setup.main(["--recover", "--reference", str(reference)]) == 0
    evidence = json.loads(reference.read_text())
    assert evidence["original_run_ids"] == ["failed-run", "old-run"]
    assert evidence["baseline_kind"] == "recovery"
    assert evidence["baseline_captured_at"]
    assert evidence["run_id"] == "test-run"
    client.get_experiment_by_name.assert_called_once_with(setup.EXPERIMENT)
    client.create_experiment.assert_not_called()
    setup.mlflow.start_run.assert_called_once_with(
        experiment_id="partial-experiment", run_name="persistence-check"
    )
    assert client.search_runs.call_args_list  # A new inventory was captured.
    client.search_registered_models.assert_called_once_with(
        filter_string=f"name = '{setup.MODEL}'", max_results=1
    )
    client.create_registered_model.assert_called_once_with(setup.MODEL)
    setup.mlflow.register_model.assert_called_once_with("models:/m-test", setup.MODEL)
    # Only these client calls may write: creation of the dedicated registry name.
    assert {call[0] for call in client.mock_calls} <= {
        "get_experiment_by_name", "search_registered_models", "search_experiments",
        "search_runs", "create_registered_model",
    }


@pytest.mark.parametrize("state", ["missing", "deleted", "model", "reference"])
def test_recovery_refuses_unsafe_or_incomplete_preconditions(offline, state):
    client, reference = offline
    client.get_experiment_by_name.return_value = NS(
        experiment_id="partial-experiment", lifecycle_stage="active"
    )
    if state == "missing":
        client.get_experiment_by_name.return_value = None
    elif state == "deleted":
        client.get_experiment_by_name.return_value.lifecycle_stage = "deleted"
    elif state == "model":
        client.search_registered_models.return_value = Page([NS(name=setup.MODEL)])
    else:
        reference.write_text("preserved evidence")
    with pytest.raises(RuntimeError):
        setup.setup(reference, recover=True)
    setup.mlflow.start_run.assert_not_called()
    setup.mlflow.register_model.assert_not_called()
    client.create_experiment.assert_not_called()
    client.create_registered_model.assert_not_called()
    if state == "reference":
        assert reference.read_text() == "preserved evidence"


def test_local_skops_failure_and_reviewed_tree_fix(tmp_path, monkeypatch):
    """Exercise real serialization and reload, using only local files."""
    from mlflow.exceptions import MlflowException
    from sklearn.tree import DecisionTreeClassifier

    # Fail immediately if this local-only test tries to create a remote client.
    monkeypatch.setattr(setup.mlflow, "MlflowClient", MagicMock(
        side_effect=AssertionError("Serialization test must remain offline")
    ))
    tree = DecisionTreeClassifier(max_depth=1, random_state=42).fit(
        [[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1]
    )
    requirements = ["scikit-learn", "skops", "numpy"]
    with pytest.raises(MlflowException, match="untrusted types"):
        setup.mlflow.sklearn.save_model(
            tree, str(tmp_path / "untrusted"), pip_requirements=requirements
        )
    trusted = tmp_path / "trusted"
    setup.mlflow.sklearn.save_model(
        tree, str(trusted), skops_trusted_types=setup.TREE_TRUSTED_TYPES,
        pip_requirements=requirements,
    )
    loaded = setup.mlflow.sklearn.load_model(str(trusted))
    np.testing.assert_array_equal(loaded.predict([[0.5], [2.5]]), [0, 1])
