# PBI-13: Robust MLflow Architecture

## 1. Original architecture and limitations

The original MLflow server ran in a Docker container on AWS Elastic Beanstalk.
Experiment metadata used SQLite (`/mlflow/mlflow.db`), while artifacts were stored
locally under `/mlflow/artifacts`. Both locations were inside the container, with
no external persistent storage configured. Container replacement or redeployment
could therefore lose experiment history, registered models and artifact files.

## 2. GenAI consultation

Four prompts were used out of the permitted nine. The consultation covered:

1. Evaluating persistent storage alternatives to container-local SQLite and files.
2. Recommending RDS MySQL and S3 in the same AWS account as the deployment.
3. Reviewing deployment configuration, environment variables and security limitations.
4. Planning independent persistence checks for runs, artifacts and registered-model inference.

These are topic summaries, not verbatim prompt transcripts. The team supplied the
working deployment files; GenAI consultation supported evaluation, review and testing.

**Consultation chat link:** [https://chatgpt.com/share/6b0ad0c7-28b4-83ec-a5c4-2a9d29c73166]

## 3. Implemented architecture

The team reports an Elastic Beanstalk deployment using a `t3.medium` instance.
The startup script defaults to one MLflow worker, limiting memory use for the
assignment workload. MLflow is pinned to version `3.18.0`.

- **Metadata:** a separate `mlflow` database on the existing RDS MySQL instance
  stores experiments, runs, metrics, parameters and model-registry metadata.
  This separates MLflow tables from the application's `spbuy` database while
  reusing the existing database service.
- **Artifacts:** S3 stores models, plots and other files, normally under the
  bucket's `artifacts` prefix. Clients upload through MLflow's artifact proxy;
  the server uses its instance profile, so laptops do not require AWS credentials.
- **Custom RDS parameter group:** The `spbuy-mysql84`
  parameter group sets `log_bin_trust_function_creators = 1` 
  to allow MLflow to initialise its database tables when
  binary logging is enabled and elevated MySQL privileges
  are unavailable. This setting applies to the entire
  RDS instance.
  `DB_PASSWORD`, optional `DB_PORT`, and `MLFLOW_DB_NAME` (default `mlflow`).
  `MLFLOW_S3_BUCKET` selects artifact storage; `MLFLOW_ARTIFACTS_DESTINATION` can
  override it. Clients continue to use `MLFLOW_TRACKING_URI`. Credentials are
  absent from the supplied deployment source and ZIP.

Moving metadata and artifacts outside the container allows them to outlive
container replacement. RDS provides shared metadata storage, and S3 provides
durable artifact storage independently of the application instance. These changes
improve persistence without requiring a change to the existing training workflow.

## 4. Testing and evidence

| Evidence | Result and attribution |
| --- | --- |
| Teammate persistence test | The teammate reported **18 runs** surviving container redeployment and a Learner Lab restart. This was not independently repeated by Codex. |
| Deployment package review | Codex compared the v3 ZIP with the Dockerfile, startup script and nginx configuration; all three entries matched their source files byte for byte. |
| Offline automated tests | Codex reported **28 passing tests**, including mocked MLflow operations and real local model serialization/reloading. These tests do not establish AWS persistence. |
| Independent live pre-restart test | The user reported **all 27 checks passing**: accessibility of **21 baseline run IDs**, test-run accessibility/status, parameters, metrics, text-artifact retrieval/content, the exact registered model version and model predictions. Codex did not execute these live checks. |
| Independent post-restart test | **Pending:** the Learner Lab is currently unavailable. Independent post-restart verification and historical artifact retrieval have not been demonstrated. |

[`persistence_setup.py`](../deploy_mlflow_aws/persistence_setup.py) creates the
dedicated `pbi13-persistence-test` experiment and registers a deterministic decision
tree as `pbi13-persistence-test-model`. It saves identifiers and expected results
in a Git-ignored local JSON reference.
[`persistence_verify.py`](../deploy_mlflow_aws/persistence_verify.py) checks that
reference and reports PASS/FAIL for each check, loading the exact model version.

The initial setup failed during model serialization: the default `skops` format
treated `sklearn.tree._tree.Tree` as untrusted. The logged-model record could
appear before model logging completed, and the run became FAILED. Local testing
confirmed the issue; explicitly trusting this reviewed built-in type resolved it.
The `--recover` option preserves the failed run, reuses the dedicated experiment,
creates a new run and captures a fresh baseline. It refuses to reuse an existing
registered-model entry and never changes champion/challenger aliases. The recovery
baseline establishes evidence from recovery onward, not from the failed attempt.

## 5. Remaining limitations

- The single-instance deployment is **not highly available**; persistent storage
  does not eliminate application downtime.
- HTTP transport, wildcard allowed hosts/CORS and the absence of authentication
  configuration in the supplied deployment files remain security concerns.
- Sharing RDS with the application can cause CPU, memory, connection and I/O contention.
- nginx retains `client_max_body_size 100M` (approximately 100 MiB), limiting uploads.
- Code review identified a conditional credential-encoding bug: `quote_plus()`
  converts spaces in usernames/passwords to `+`, which SQLAlchemy interprets as a
  literal plus sign. Credentials containing spaces may fail; the supplied startup
  script remains unchanged.

The teammate's reported restart result and the independent pre-restart checks
support the design, but independent post-restart evidence remains outstanding.
