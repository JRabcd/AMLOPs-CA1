import mlflow

# Replace with your actual AWS domain
mlflow.set_tracking_uri(
    "http://mlflow-aws-eb.eba-x8mdpp5p.us-east-1.elasticbeanstalk.com/"
)

mlflow.set_experiment("Sprint 3 Test")

with mlflow.start_run(run_name="PBI-06 Deployment Verification"):
    mlflow.log_param("model", "test_model")
    mlflow.log_metric("accuracy", 0.95)

print("Experiment logged successfully!")
