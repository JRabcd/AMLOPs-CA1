# SP-buy Fraud Detection (ST1519 Agile MLOps CA1)

SP-buy's Trust & Safety team reviews suspicious orders by hand and with a few hard-coded rules.
This project builds an automated pipeline that cleans new order data, loads it into AWS RDS,
retrains fraud-detection models and tracks them in MLflow, so analysts can focus on the orders
that are most likely to be fraud.

- **Product backlog and sprint board:** [GitHub Project](https://github.com/users/JRabcd/projects/1)
- **Team (Group 3):** Qian Yunyang (`qyyqianyy`), Jian Rong (`JRabcd`), Zheng Hao (`lzh1625`)

## Repository layout

| Folder | What goes here |
|---|---|
| `data/` | Local copies of the CSV batches. **Not committed** – see [`data/README.md`](data/README.md). |
| `notebooks/` | Jupyter notebooks for exploration and experiments (EDA, PBI-02). |
| `src/` | Reusable Python code: cleaning, transformation, ETL, training. |
| `sql/` | Database schema scripts for AWS RDS (MySQL). |
| `docs/` | Design notes, e.g. the RDS schema design. |

## Getting started

1. Clone the repo and create a virtual environment:
   ```bash
   git clone https://github.com/JRabcd/AMLOPs-CA1.git
   cd AMLOPs-CA1
   python -m venv .venv
   ```
   Activate it with `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (macOS/Linux).
2. Install the project and its runtime dependencies from the repository root:
   ```bash
   python -m pip install -e .
   ```
   This makes the local `src` package importable, including when running scripts in
   `deploy_mlflow_aws/`. Do not run `pip install src`: `src` is this repository's
   code, not an external dependency. For the plotting and notebook tools as well,
   run `python -m pip install -r requirements.txt`.
   Put the data files in `data/` (see [`data/README.md`](data/README.md)).
3. Resource settings (RDS host, database credentials, MLflow URL) are read from environment
   variables or `.env`. Copy the `.env.example` template to `.env`
   and fill in your own values. `.env` is ignored by git.

## Running Model C

Use the project virtual environment (in VS Code, select `.venv/Scripts/python.exe`
as the Python interpreter). From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip check
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe -m src.train_memberC --batch base
```

The PBI-07C implementation is `src/train_memberC.py`; `modelC_test.py` calls it
to verify real model uploads to Elastic Beanstalk. Configure `MLFLOW_TRACKING_URI`
and `DATA_DIR` in `.env` or your shell before running. See [the PBI-07C run guide](docs/pbi07c.md)
for local and AWS commands, verified results, and the v1-to-v2 diagnosis.

## Loading a new data batch

```bash
python -m src.etl.run_etl --source <url-or-folder> --batch <postfix>
```

Reads `customers_<postfix>.csv`, `orders_<postfix>.csv` and `labels_<postfix>.csv`, cleans them and upserts them into RDS. See [`docs/etl.md`](docs/etl.md).

## How we work

- **One issue per task.** Every task on the board is a GitHub issue with one owner.
- **Branch per task:** `feature/pbi-<number>-<short-name>`, e.g. `feature/pbi-05-rds-schema`.
- **Pull requests into `main`.** Put `Closes #<issue>` in the PR description so merging closes the
  task and moves it to *Complete* on the board.
- **One review before merge.** Another member reviews and approves every pull request.
- **No secrets or data in git.** Credentials go in `.env`; CSV files stay in `data/` locally.
