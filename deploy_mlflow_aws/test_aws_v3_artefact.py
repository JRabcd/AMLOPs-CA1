"""Verify artifact uploads: python -m deploy_mlflow_aws.test_aws_v3_artefact."""

import os
import tempfile

import mlflow

from src.db.config import load_dotenv


def main() -> None:
    load_dotenv()
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("Set MLFLOW_TRACKING_URI (environment or .env).")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("100mb limit test - 5mb")

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_artifact_5mb.bin")

        # Generate a 5 MiB dummy artifact.
        with open(file_path, "wb") as f:
            f.write(os.urandom(5 * 1024 * 1024))

        with mlflow.start_run(run_name="PBI-06 Artifact Upload Verification"):
            mlflow.log_param("artifact_size_mb", 5)
            mlflow.log_artifact(file_path)

    print("5 MiB artifact uploaded successfully!")


if __name__ == "__main__":
    main()
