# Automated retraining (PBI-10)

After the ETL has loaded a batch, one command retrains every member's models on **all** data in RDS
(the base data plus every new batch) and logs the runs to MLflow:

```bash
python -m src.retrain                  # runs named after the latest successful ETL batch
python -m src.retrain --batch XXX      # or name them yourself
python -m src.retrain --members A C    # only some members' models
python -m src.retrain --no-plots       # skip the confusion matrix and PR curve images
python -m src.retrain --sequential     # members one after another (default: at the same time)
python -m src.retrain --no-cache       # download every row instead of only changed rows
```

Settings come from `.env` (copy `.env.example`): `DB_*` for RDS and `MLFLOW_TRACKING_URI` for the team MLflow
server. To try it without the server, use `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`.

## What it does

```
RDS: customers, orders, labels ──► same dtypes as cleaning ──► prepare_features (src/transform.py)
        src/retrain.py                                               │
                                                                      ▼
        train(frame, batch) in src/train_memberA.py, train_memberB.py, train_memberC.py
                                                                      │
                                                                      ▼
        MLflow runs etl_<batch>_<member>_<model>  +  summary table in the log
```

- **Same data as training on the CSVs.** Columns read from RDS get the exact dtypes `src/cleaning.py`
  produces. Checked on the base data: same 949,872 rows, same split, and model inputs within 0.00005
  (RDS stores money as `DECIMAL(12,4)`, so values are rounded to 4 decimal places).
- **Run names show the batch.** `etl_base_QianYunyang_lgbm_v1`, `etl_XXX_JianRong_...`: filter MLflow by
  `etl_XXX` to see every model from one ETL run.
- **One member's bug doesn't stop the others.** A failed member is logged and listed in the summary; the
  other members still train. The exit code is `1` if any member failed, so the pipeline (PBI-12) can tell.
- **Missing scripts are skipped**, with a warning, so retraining works before every member's script is merged.

## Adding your models (07b, 07c)

Add `src/train_memberB.py` (Jian Rong) or `src/train_memberC.py` (Zheng Hao) with the same function as
`src/train_memberA.py`:

```python
def train(frame: pd.DataFrame, batch: str = "base", plots: bool = True) -> dict[str, dict]:
    """Train and log your model versions on an already loaded frame; return {model_name: metrics}."""
    parts = split(frame)
    return {
        "xgb_v1": train_and_log("xgb_v1", build(V1), V1, make_matrices(frame, *parts), MEMBER, batch, plots),
        "xgb_v2": ...,
    }
```

Use `split`, `make_matrices` and `train_and_log` from `src/train_common.py`, so all models share the same data,
split and metrics and can be compared fairly in the model review (PBI-08). Nothing else needs changing:
`src/retrain.py` already looks for these files.

## Timing

Members train at the same time (threads) and only rows changed since the last run are downloaded from RDS.
Details and measurements are in [`docs/pipeline.md`](pipeline.md), which runs the ETL and this retraining in one command (PBI-12).
