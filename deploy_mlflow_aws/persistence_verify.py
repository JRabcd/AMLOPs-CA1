"""Check saved evidence after a restart: python -m deploy_mlflow_aws.persistence_verify."""

import argparse
import json
import math
from pathlib import Path
import tempfile

import mlflow.sklearn
import numpy as np

from deploy_mlflow_aws.persistence_setup import MODEL, REFERENCE, configure_tracking


def verify(reference: Path = REFERENCE) -> bool:
    evidence = json.loads(reference.read_text(encoding="utf-8"))
    if evidence["schema_version"] != 1 or evidence["model_name"] != MODEL:
        raise ValueError("Unsupported persistence reference")
    client = configure_tracking()
    results = []

    def check(label, action):
        try:
            action()
        except Exception as exc:
            print(f"FAIL: {label} ({type(exc).__name__})")
            results.append(False)
        else:
            print(f"PASS: {label}")
            results.append(True)

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    for run_id in evidence["original_run_ids"]:
        def original_run(run_id=run_id):
            require(client.get_run(run_id).info.run_id == run_id, "Run ID mismatch")
        check(f"original run {run_id} accessible", original_run)
    if not evidence["original_run_ids"]:
        print("PASS: original run inventory (0 runs recorded)")

    def test_run():
        run = client.get_run(evidence["run_id"])
        require(run.info.experiment_id == evidence["experiment_id"], "Experiment mismatch")
        require(run.info.status == "FINISHED", "Test run not finished")
    check("test run accessible and finished", test_run)

    def parameters():
        run = client.get_run(evidence["run_id"])
        for key, expected in evidence["params"].items():
            require(run.data.params.get(key) == expected, "Parameter mismatch")
    check("test run parameters", parameters)

    def metrics():
        run = client.get_run(evidence["run_id"])
        for key, expected in evidence["metrics"].items():
            require(math.isclose(run.data.metrics[key], expected, rel_tol=1e-9, abs_tol=1e-12),
                    "Metric mismatch")
    check("test run metrics", metrics)

    def artifact():
        with tempfile.TemporaryDirectory() as directory:
            downloaded = client.download_artifacts(
                evidence["run_id"], evidence["artifact_path"], directory
            )
            require(Path(downloaded).read_text(encoding="utf-8") == evidence["artifact_text"],
                    "Artifact contents mismatch")
    check("text artifact contents", artifact)

    def registered_version():
        version = client.get_model_version(MODEL, evidence["model_version"])
        require(str(version.version) == evidence["model_version"], "Version mismatch")
        require(version.run_id == evidence["run_id"], "Model run mismatch")
    check("exact registered model version", registered_version)

    def predictions():
        model = mlflow.sklearn.load_model(f"models:/{MODEL}/{evidence['model_version']}")
        actual = model.predict(np.asarray(evidence["prediction_inputs"]))
        require(np.array_equal(actual, evidence["expected_predictions"]), "Predictions mismatch")
    check("registered model predictions", predictions)
    return all(results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args(argv)
    try:
        return 0 if verify(args.reference) else 1
    except Exception as exc:
        print(f"FAIL: verification prerequisites ({type(exc).__name__}); "
              "check the reference file, MLFLOW_TRACKING_URI and server access.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
