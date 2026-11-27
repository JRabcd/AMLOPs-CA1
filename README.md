# SP-buy Fraud Detection (ST1519 Agile MLOps CA1)

SP-buy's Trust & Safety team reviews suspicious orders by hand and with a few hard-coded rules.
This project builds an automated pipeline that cleans new order data, loads it into AWS RDS,
retrains fraud-detection models and tracks them in MLflow, so analysts can focus on the orders
that are most likely to be fraud.

- **Product backlog and sprint board:** [GitHub Project](https://github.com/users/JRabcd/projects/1)
- **IT infrastructure managers:** set-up, configuration and running the pipeline: [`docs/it-manager-guide.md`](docs/it-manager-guide.md)
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
2. Put the data files in `data/` (see [`data/README.md`](data/README.md)).
3. Resource settings (RDS host, database credentials, MLflow URL) are read from environment
   variables, never hard-coded. A template `.env.example` is added under PBI-11; copy it to `.env`
   and fill in your own values. `.env` is ignored by git.

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
