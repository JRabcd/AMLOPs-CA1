# One-command pipeline (PBI-12)

When the operations team publishes a new batch, one command loads it and retrains every model:

```bash
# batch published as <prefix>/customers_XXX.csv, orders_XXX.csv, labels_XXX.csv
python -m src.run_pipeline --source https://example.com/data --batch XXX

# or the three files explicitly
python -m src.run_pipeline --customers URL --orders URL --labels URL
```

Settings come from `.env` (copy `.env.example`): `DB_*` for RDS and `MLFLOW_TRACKING_URI` for MLflow.

| Option | Effect |
|---|---|
| `--batch-id NAME` | Record the batch under another name |
| `--force` | Reload a batch that is already in RDS |
| `--members A C` | Train only some members' models |
| `--no-plots` | Skip the confusion matrix and PR curve images |
| `--sequential` | Train members one after another instead of at the same time |
| `--no-cache` | Download every row from RDS instead of only the changed ones |

## What it does

| Step | What happens | Code |
|---|---|---|
| 1. check | Database and MLflow settings are checked first, so a typo fails in seconds | `src/db/config.py`, `src/train_common.py` |
| 2. ETL | Download, clean and load the three CSVs into RDS; recorded in `etl_batches` | `src/etl/run_etl.py` |
| 3. read | All data in RDS, old and new. Only rows changed since the last run are downloaded | `src/rds_cache.py` |
| 4. train | Every member's models at the same time, logged as `etl_<batch>_<member>_<model>` | `src/retrain.py` |

The log ends with a summary: rows loaded, every model's test PR-AUC, precision, recall and F1, how long each
step took, and whether the run finished within 3 minutes.

**Errors:** if the ETL fails (bad URL, wrong columns, database down), nothing is retrained and the exit code is 1.
If one member's training fails, the other members still train and log, the summary lists it as FAILED, and
the exit code is 1.

## Keeping it under 3 minutes

The demo rule is that each pipeline finishes within 3 minutes. Measured on 26–27 Nov with the base data and all
three members' model types:

| Change | Why it was slow | Effect |
|---|---|---|
| Members train at the same time | Most of the time is uploading models to MLflow (~2 MB each). The server is in the US, so one upload runs at about 230 KB/s, but six uploads at once take no longer than one | Training + logging 199 s → 124 s |
| Package list given to MLflow | Without it, MLflow starts a separate Python process per model to work out its requirements | ~8 s saved per model |
| Feature matrices built once | Members with the same inputs (all but LightGBM v2) used to build them separately | ~5 s saved per member |
| Only changed rows read from RDS | Reading all 2.8 million rows took 25 s on the school network and 8 minutes on a home network | Each run downloads only the new batch |
| Index on `loaded_at` (`sql/migrations/002_index_loaded_at.sql`) | Without it, finding the changed rows scans every table | Read with nothing new: 23 s → 7 s (home network) |

**Before the demo:** run the pipeline (or `python -m src.retrain`) once on the laptop you will demo from, so
the local copy of the RDS data (`.cache/rds/`, git-ignored) exists. On the day, only the tutor's new batch is
then downloaded. If the copy is missing or doesn't match RDS, the pipeline downloads everything, which is
correct but slower.

The upload speed to the MLflow server depends on the network you are on, so time a run on the same network
you will demo from (PBI-12.3/12.4).

## Database change

`sql/migrations/002_index_loaded_at.sql` adds an index on `loaded_at` to customers, orders and labels (applied to the team database on 27 Nov with `python -m src.db.apply_schema`; safe to run again). New databases get the index from `sql/schema.sql`.
