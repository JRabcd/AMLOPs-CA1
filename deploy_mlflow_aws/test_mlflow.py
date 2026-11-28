"""Verify MLflow logging: python -m deploy_mlflow_aws.test_mlflow."""

import os

import mlflow

from src.db.config import load_dotenv


def main() -> None:
    load_dotenv()
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("Set MLFLOW_TRACKING_URI (environment or .env).")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("Sprint 3 Test")

    with mlflow.start_run(run_name="PBI-06 Deployment Verification"):
        mlflow.log_param("model", "test_model")
        mlflow.log_metric("accuracy", 0.95)

    print("Experiment logged successfully!")


if __name__ == "__main__":
    main()
