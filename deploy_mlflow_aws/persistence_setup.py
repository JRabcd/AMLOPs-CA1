"""Create persistence evidence: python -m deploy_mlflow_aws.persistence_setup."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from urllib.parse import quote, quote_plus

import mlflow
import mlflow.sklearn
from mlflow.entities import ViewType
from mlflow.models import infer_signature
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.db.config import load_dotenv


EXPERIMENT = "pbi13-persistence-test"
MODEL = "pbi13-persistence-test-model"
REFERENCE = Path(__file__).with_name("persistence_reference.json")
# Reviewed sklearn implementation type used by our own deterministic tree.
TREE_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def redact_diagnostic(message):
    """Strip environment secrets, URI contents and common credential fields."""
    text = str(message)
    secrets = set()
    for key, value in os.environ.items():
        if value and re.search(r"PASSWORD|SECRET|TOKEN|CREDENTIAL|AUTH|ACCESS_KEY|PRIVATE_KEY|DB_USER", key, re.I):
            secrets.update((value, quote(value, safe=""), quote_plus(value, safe=""),
                            repr(value)[1:-1], json.dumps(value)[1:-1]))
    for value in sorted(secrets, key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    text = re.sub(r"[a-zA-Z][\w+.-]*://[^\s<>\"']+", "[URI REDACTED]", text)
    text = re.sub(r"(?is)\[parameters:.*?\](?=\s|$)", "[parameters: REDACTED]", text)
    text = re.sub(r"(?i)(authorization\s*[:=]\s*)(?:Bearer|Basic)\s+[^\s,;}]+",
                  r"\1[REDACTED]", text)
    text = re.sub(
        r"(?i)([\"']?(?:password|passwd|pwd|secret|token|authorization|"
        r"aws_access_key_id|aws_secret_access_key|x-amz-security-token)[\"']?\s*[:=]\s*)"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)",
        r"\1[REDACTED]", text,
    )
    text = re.sub(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", "[REDACTED]", text)
    return text


class SetupStepError(RuntimeError):
    def __init__(self, stage):
        self.stage = stage
        super().__init__(f"Setup failed during: {stage}")


@contextmanager
def setup_step(stage):
    print(f"STEP: {stage}", flush=True)
    try:
        yield
    except SetupStepError:
        raise
    except Exception as exc:
        raise SetupStepError(stage) from exc


def report_error(exc):
    """Show the exception chain without traceback source lines, locals or URIs."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        code = getattr(exc, "error_code", None)
        detail = f"{type(exc).__name__}: {exc}"
        if code:
            detail += f" [error_code={code}]"
        print(redact_diagnostic(detail))
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)


def configure_tracking():
    load_dotenv()
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("Set MLFLOW_TRACKING_URI (environment or .env).")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)


def existing_run_ids(client):
    """Include all experiments/runs and follow both APIs' pagination tokens."""
    result = []
    experiment_token = None
    while True:
        experiments = client.search_experiments(
            view_type=ViewType.ALL, max_results=1000, page_token=experiment_token
        )
        for experiment in experiments:
            run_token = None
            while True:
                runs = client.search_runs(
                    [experiment.experiment_id], run_view_type=ViewType.ALL,
                    max_results=1000, page_token=run_token,
                )
                result.extend(run.info.run_id for run in runs)
                run_token = runs.token
                if not run_token:
                    break
        experiment_token = experiments.token
        if not experiment_token:
            return sorted(set(result))


def setup(reference: Path = REFERENCE, *, recover: bool = False):
    if reference.exists():
        raise RuntimeError(f"Reference already exists: {reference}. Keep it for verification.")
    if mlflow.active_run() is not None or os.environ.get("MLFLOW_RUN_ID"):
        raise RuntimeError("End the active MLflow run and unset MLFLOW_RUN_ID before setup.")
    with setup_step("configure tracking"):
        client = configure_tracking()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if recover:
        if experiment is None or experiment.lifecycle_stage != "active":
            raise RuntimeError(f"Recovery requires the existing active experiment {EXPERIMENT}.")
    elif experiment is not None:
        raise RuntimeError(f"Experiment {EXPERIMENT} already exists; refusing to modify it.")
    # Even recovery cannot append versions to any existing registered model.
    if client.search_registered_models(filter_string=f"name = '{MODEL}'", max_results=1):
        raise RuntimeError(f"Registered model {MODEL} already exists; refusing to modify it.")
    with setup_step("record original run IDs"):
        baseline_captured_at = datetime.now(timezone.utc).isoformat()
        original_runs = existing_run_ids(client)
    if recover:
        experiment_id = experiment.experiment_id
    else:
        with setup_step("create test experiment"):
            experiment_id = client.create_experiment(EXPERIMENT)
    print(f"Experiment ID: {redact_diagnostic(experiment_id)}", flush=True)

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X, y)
    inputs = np.array([[0.5], [2.5]])
    artifact_path = "persistence.txt"
    artifact_text = "PBI-13.4 persistence evidence\n"
    with setup_step("start or finish test run"):
        with mlflow.start_run(experiment_id=experiment_id, run_name="persistence-check") as run:
            run_id = run.info.run_id
            print(f"Run ID: {redact_diagnostic(run_id)}", flush=True)
            with setup_step("log test parameter"):
                mlflow.log_param("purpose", "persistence")
            with setup_step("log test metric"):
                mlflow.log_metric("accuracy", 1.0)
            with setup_step("log text artifact"):
                mlflow.log_text(artifact_text, artifact_path)
            with setup_step("serialize and upload decision tree model"):
                logged = mlflow.sklearn.log_model(
                    model, name="model", signature=infer_signature(X, model.predict(X)),
                    input_example=X[:1], skops_trusted_types=TREE_TRUSTED_TYPES,
                )

    # Explicit creation fails if another process created the name meanwhile.
    with setup_step("create registered model"):
        client.create_registered_model(MODEL)
    with setup_step("register model version"):
        version = mlflow.register_model(logged.model_uri, MODEL)
    evidence = {
        "schema_version": 1,
        "baseline_kind": "recovery" if recover else "initial",
        "baseline_captured_at": baseline_captured_at,
        "original_run_ids": original_runs,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "params": {"purpose": "persistence"},
        "metrics": {"accuracy": 1.0},
        "artifact_path": artifact_path,
        "artifact_text": artifact_text,
        "model_name": MODEL,
        "model_version": str(version.version),
        "model_uri": logged.model_uri,
        "prediction_inputs": inputs.tolist(),
        "expected_predictions": model.predict(inputs).tolist(),
    }
    with setup_step("save local reference"):
        with reference.open("x", encoding="utf-8") as handle:
            json.dump(evidence, handle, indent=2)
            handle.write("\n")
    print(f"PASS: setup complete; reference saved to {reference}")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument(
        "--recover", action="store_true",
        help="Reuse the dedicated active test experiment, preserve old runs, and capture a fresh baseline.",
    )
    args = parser.parse_args(argv)
    try:
        setup(args.reference, recover=args.recover)
    except Exception as exc:
        print("FAIL: setup; diagnostic details (credentials redacted):")
        report_error(exc)
        print("Remote objects may be partially created. Preserve them and inspect the failed "
              "stage before attempting recovery; do not rerun setup blindly.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
