import mlflow
import os
import tempfile

mlflow.set_tracking_uri(
    "http://mlflow-aws-eb.eba-x8mdpp5p.us-east-1.elasticbeanstalk.com/"
)

mlflow.set_experiment("100mb limit test - 5mb")

with tempfile.TemporaryDirectory() as temp_dir:
    file_path = os.path.join(temp_dir, "test_artifact_5mb.bin")

    # Generate a 50 MiB dummy artifact
    with open(file_path, "wb") as f:
        f.write(os.urandom(5 * 1024 * 1024))

    with mlflow.start_run(run_name="PBI-06 Artifact Upload Verification"):
        mlflow.log_param("artifact_size_mb", 5)
        mlflow.log_artifact(file_path)

print("25 MB artifact uploaded successfully!")