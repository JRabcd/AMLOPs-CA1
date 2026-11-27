# Guide for IT infrastructure managers (PBI-14)

How to set up the database, configure the code and run the data pipeline without the project team.
Each service is set up with one command once its settings are in place.

| You want to... | Section |
|---|---|
| Create the database and its tables | [1. RDS database](#1-rds-database) |
| Change a host, password or the MLflow address | [2. Configuration](#2-configuration) |
| Load a new data batch and retrain the models | [3. Running the pipeline](#3-running-the-pipeline) |
| Fix an error | [4. Troubleshooting](#4-troubleshooting) |
| Redeploy the MLflow server | [5. MLflow server](#5-mlflow-server) |

## 0. One-time set-up on the machine that runs the pipeline

Python 3 (the team tests with 3.14), then from the repository folder:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 1. RDS database

### Create the instance (AWS console)

1. **RDS → Create database → Standard create → MySQL** (the team uses MySQL 8.4).
2. **Templates:** Free tier or Dev/Test. **Instance:** `db.t3.micro` is enough.
3. **Settings:** pick a DB instance identifier (the team's is `spbuy-db`), a master username and a master
   password. Keep the password out of email, chat history and git.
4. **Connectivity:** *Public access = Yes*, so the team's laptops and the MLflow server can reach it.
   In the VPC security group, allow inbound **MySQL/Aurora, TCP 3306** from the addresses that need it
   (the team currently allows any address, because members connect from changing home and school networks;
   narrow it if you can).
5. Create it, wait until the status is **Available**, and copy the **endpoint** from *Connectivity & security*.

### Create the tables (one command)

Fill in `.env` (next section), then:

```bash
python -m src.db.apply_schema
```

This creates the `spbuy` database and the `customers`, `orders`, `labels` and `etl_batches` tables from
`sql/schema.sql`, then applies everything in `sql/migrations/`. It is safe to run again: nothing that
exists is dropped. Table design: [`docs/rds-schema.md`](rds-schema.md).

### Check it works

```bash
python -m src.db.check_connection
```

The last line should be `OK: the database is ready to use.` If not, it says what to fix (see section 4).

### Load the historical data (once)

```bash
python -m src.etl.run_etl --source <folder or URL with the base CSVs> --batch base
```

About 2.8 million rows; this took about 6 minutes on a fast network.

## 2. Configuration

No address or password is written in the code. Every script reads these settings from environment
variables, or from a file called `.env` in the repository folder (environment variables win if both are set).
Start from the template:

```bash
copy .env.example .env            # macOS/Linux: cp .env.example .env
```

| Setting | Required | What it is | Example |
|---|---|---|---|
| `DB_HOST` | yes | RDS endpoint | `spbuy-db.xxxxxxxx.us-east-1.rds.amazonaws.com` |
| `DB_PORT` | no | MySQL port, default `3306` | `3306` |
| `DB_USER` | yes | Database user | `admin` |
| `DB_PASSWORD` | yes | That user's password | |
| `DB_NAME` | no | Database name, default `spbuy` | `spbuy` |
| `MLFLOW_TRACKING_URI` | for training | Address of the MLflow server | `http://<environment>.elasticbeanstalk.com` |

To change a setting, edit `.env` (or the environment variable) and run the command again; no code changes are
needed. `.env` is in `.gitignore`, so it is never committed. Share it with team members privately.

**When the password changes:** update `DB_PASSWORD` in every `.env` that uses it, then run
`python -m src.db.check_connection` to confirm.

## 3. Running the pipeline

### When a new batch arrives

The operations team publishes three files with the same postfix, e.g. `customers_XXX.csv`, `orders_XXX.csv`
and `labels_XXX.csv`. One command loads them and retrains every model:

```bash
python -m src.run_pipeline --source <URL prefix> --batch XXX
# or the three files explicitly
python -m src.run_pipeline --customers <URL> --orders <URL> --labels <URL>
```

It checks the settings, loads the batch into RDS, reads all data (old and new) from RDS, retrains every
team member's models and logs them to MLflow. Details: [`docs/pipeline.md`](pipeline.md).

### Where the results are

| What | Where |
|---|---|
| Rows loaded, model scores, time per step | The summary at the end of the command's output |
| Every load, its row counts and any error | The `etl_batches` table in RDS |
| Models, metrics and plots | MLflow, experiment `spbuy-fraud`; runs are named `etl_<batch>_<member>_<model>`, so search for `etl_XXX` to see one batch's runs |

The exit code is `0` if everything succeeded and `1` if any step failed, so the command can be scheduled.

### Running one step on its own

| Step | Command | Docs |
|---|---|---|
| Load a batch only | `python -m src.etl.run_etl --source <URL prefix> --batch XXX` | [`docs/etl.md`](etl.md) |
| Read a batch without writing to RDS | add `--dry-run` to the command above | |
| Retrain only (data already in RDS) | `python -m src.retrain` | [`docs/retraining.md`](retraining.md) |

### Keeping it fast

The first retraining run on a machine downloads every row from RDS and keeps a copy in `.cache/rds/`; later
runs download only rows that changed. Run `python -m src.retrain` once on a new machine before a timed run.
Delete `.cache/` (or add `--no-cache`) to force a full download.

## 4. Troubleshooting

| Message | Likely cause | Fix |
|---|---|---|
| `Missing database settings: DB_HOST ...` | `.env` missing or incomplete | Copy `.env.example` to `.env` and fill it in |
| `Can't reach the server` / error 2003 | AWS Learner Lab stopped, RDS not *Available*, or port 3306 blocked | Start the lab, wait for *Available*, check the security group; some networks block 3306 |
| `Wrong user or password` / error 1045 | Wrong `DB_USER` or `DB_PASSWORD` | Correct `.env` |
| `The database doesn't exist yet` / error 1049 | Tables not created | `python -m src.db.apply_schema` |
| `Lost connection to MySQL server during query` | Very slow network | Run again; the retraining connection already allows 10 minutes |
| `Set MLFLOW_TRACKING_URI ...` | MLflow address missing | Add it to `.env` |
| MLflow connection refused or times out | MLflow server's lab or Elastic Beanstalk environment is stopped | Start it; check the URL in `.env` |
| `HTTP 404` or `File not found` in the ETL | Wrong URL or postfix | Check the three file names |
| `missing column(s)` in the ETL | A file doesn't match the expected format | Ask the operations team; the batch is not loaded |
| `FAILED <member>` in the summary | That member's training script raised an error | The other members' models are still logged; the log shows the error |

## 5. MLflow server

The MLflow server runs as a Docker container on AWS Elastic Beanstalk. The deployment files are in
`deploy_mlflow_aws/` (`Dockerfile`, the nginx upload limit in `.platform/`, and the zip uploaded to Elastic
Beanstalk). The team is moving its storage to RDS and S3 (PBI-13) so that runs survive restarts; the
step-by-step redeployment instructions will be added here once that is final (PBI-14.1).
