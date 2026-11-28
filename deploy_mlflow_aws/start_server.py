"""Start the team MLflow server with permanent storage (PBI-13).

Run records (experiments, parameters, metrics, registered models) go into a MySQL database called `mlflow`
on the team's RDS instance, and model files and plots go into an S3 bucket. Both outlive the Elastic
Beanstalk instance, so a restart of the Learner Lab no longer loses runs.

Settings come from environment variables. On Elastic Beanstalk they are environment properties
(Configuration -> Updates, monitoring and logging -> Environment properties), so the password never
goes into this repository:

    DB_HOST, DB_USER, DB_PASSWORD   the team RDS instance (the same values as in .env)
    DB_PORT                         optional, default 3306
    MLFLOW_DB_NAME                  optional, default mlflow (a separate database from spbuy)
    MLFLOW_S3_BUCKET                the S3 bucket for model files
    MLFLOW_ARTIFACTS_DESTINATION    optional, overrides the S3 location (e.g. a local folder for testing)

Clients upload model files through this server (MLflow's artifact proxy), so team laptops need no AWS
credentials. The server itself reaches S3 through the instance profile (LabInstanceProfile in the
Learner Lab).
"""
import os
import sys
from urllib.parse import quote_plus

import pymysql


def setting(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if not value:
        sys.exit(f"Missing environment property {name}. Set it in Elastic Beanstalk -> Configuration -> "
                 "Updates, monitoring and logging -> Environment properties.")
    return value


host = setting("DB_HOST")
port = int(setting("DB_PORT", "3306"))
user = setting("DB_USER")
password = setting("DB_PASSWORD")
db_name = setting("MLFLOW_DB_NAME", "mlflow")
artifacts = os.environ.get("MLFLOW_ARTIFACTS_DESTINATION") or f"s3://{setting('MLFLOW_S3_BUCKET')}/artifacts"

if not db_name.replace("_", "").isalnum():
    sys.exit(f"MLFLOW_DB_NAME must be letters, digits or _ only, not {db_name!r}")

# MLflow creates its own tables, but the database itself must exist first.
conn = pymysql.connect(host=host, port=port, user=user, password=password, connect_timeout=15)
with conn.cursor() as cur:
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
conn.close()
print(f"MLflow storage: database {db_name} on {host}, files in {artifacts}", flush=True)

# Passed as environment variables, not command-line arguments, so the password stays out of the process list.
os.environ["MLFLOW_BACKEND_STORE_URI"] = (
    f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db_name}")
os.environ["MLFLOW_ARTIFACTS_DESTINATION"] = artifacts
command = [
    sys.executable, "-m", "mlflow", "server",
    "--host", "0.0.0.0",
    "--port", os.environ.get("MLFLOW_PORT", "5000"),
    "--workers", os.environ.get("MLFLOW_WORKERS", "1"),  # each worker needs ~300 MB; one is plenty for the team
    "--allowed-hosts", "*",
    "--cors-allowed-origins", "*",
]
if os.name == "nt":  # local try-out on Windows, where exec doesn't replace the process
    import subprocess
    sys.exit(subprocess.call(command))
os.execv(sys.executable, command)  # in Docker: MLflow replaces this process, so it receives stop signals directly
