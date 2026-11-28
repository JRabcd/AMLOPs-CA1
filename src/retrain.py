"""Retrain every member's models on all data in RDS and log them to MLflow (PBI-10).

    python -m src.retrain                   # batch name = latest successful ETL batch
    python -m src.retrain --batch XXX       # name the runs etl_XXX_<member>_<model>
    python -m src.retrain --members A C     # only some members

Steps: read the cleaned tables from RDS (old + new batches together), build features with
src/transform.py, then call train(frame, batch) in each member's training script. A member
whose script is not in the repo yet is skipped; a member whose training fails is reported and
the others still run. Exit code 0 only if every member that ran succeeded.

Settings come from .env: DB_* for RDS (src/db/config.py) and MLFLOW_TRACKING_URI for MLflow.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from dataclasses import dataclass, field

import pandas as pd
import pymysql
from pymysql.constants import FIELD_TYPE
from pymysql.converters import conversions

from src.db.config import connect
from src.etl.load import COLUMNS
from src.train_common import connect_mlflow, drop_unusable
from src.transform import prepare_features

logger = logging.getLogger("retrain")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [retrain]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# Member letter -> training module. Each module must define train(frame, batch=..., plots=...).
TRAINERS = {
    "A": ("QianYunyang", "src.train_memberA"),
    "B": ("JianRong", "src.train_memberB"),
    "C": ("ZhengHao", "src.train_memberC"),
}

# Same column types as src.cleaning.clean_batch produces, so features match training on CSVs.
STRING_COLUMNS = {"country_code", "customer_id", "order_id", "collect_type", "payment_method"}
INT_COLUMNS = {"num_orders_last_50days", "num_cancelled_orders_last_50days", "num_refund_orders_last_50days",
               "num_associated_customers", "num_items_ordered"}
FLOAT_COLUMNS = {"total_payment_last_50days", "order_value", "refund_value"}
DATE_COLUMNS = {"first_order_datetime": "%Y-%m-%d %H:%M:%S", "order_date": "%Y-%m-%d"}

# Fetch DECIMAL as float and dates as text: much faster than Decimal/date objects for ~1M rows.
_CONVERTERS = {**conversions, FIELD_TYPE.NEWDECIMAL: float, FIELD_TYPE.DECIMAL: float,
               FIELD_TYPE.DATE: str, FIELD_TYPE.DATETIME: str}


@dataclass
class RetrainResult:
    batch: str
    rows: int = 0
    seconds: float = 0.0
    runs: dict[str, dict] = field(default_factory=dict)       # "<member>/<model>" -> metrics
    failed: dict[str, str] = field(default_factory=dict)      # member -> error
    skipped: list[str] = field(default_factory=list)          # members without a script

    @property
    def ok(self) -> bool:
        return not self.failed and bool(self.runs)


def to_clean_types(df: pd.DataFrame) -> pd.DataFrame:
    """Give columns read from RDS the dtypes clean_batch() outputs."""
    out = df.copy()
    for col in out.columns:
        if col in STRING_COLUMNS:
            out[col] = out[col].astype("string")
        elif col in INT_COLUMNS:
            out[col] = pd.to_numeric(out[col]).astype("Int64")
        elif col in FLOAT_COLUMNS:
            out[col] = pd.to_numeric(out[col]).astype("float64")
        elif col in DATE_COLUMNS:
            out[col] = pd.to_datetime(out[col], format=DATE_COLUMNS[col], errors="coerce").astype("datetime64[ns]")
        elif col == "mobile_verified":
            out[col] = out[col].map({1: True, 0: False}).astype("boolean")
        elif col == "is_fraud":
            out[col] = out[col].astype("int64")
    return out


def read_table(conn, table: str) -> pd.DataFrame:
    """Every row of one cleaned table, with clean_batch() dtypes."""
    cols = COLUMNS[table]
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(cols)} FROM {table}")
        rows = cur.fetchall()
    return to_clean_types(pd.DataFrame(rows, columns=cols))


def latest_batch(conn) -> str:
    """Name of the most recent successfully loaded ETL batch."""
    with conn.cursor() as cur:
        cur.execute("SELECT batch_id FROM etl_batches WHERE status = 'success' "
                    "ORDER BY finished_at DESC, started_at DESC LIMIT 1")
        row = cur.fetchone()
    if not row:
        raise RuntimeError("No successful ETL batch in RDS yet. Run python -m src.etl.run_etl first.")
    return row[0]


def load_training_frame(batch: str | None = None) -> tuple[pd.DataFrame, str]:
    """All labelled orders in RDS as model-ready rows, plus the batch name to use for the runs."""
    t0 = time.perf_counter()
    conn = connect(conv=_CONVERTERS)
    try:
        batch = batch or latest_batch(conn)
        tables = {t: read_table(conn, t) for t in ("customers", "orders", "labels")}
    finally:
        conn.close()
    logger.info("Read from RDS in %.1fs: %s", time.perf_counter() - t0,
                ", ".join(f"{t} {len(df):,}" for t, df in tables.items()))
    frame = drop_unusable(prepare_features(tables["customers"], tables["orders"], tables["labels"]))
    logger.info("Training frame: %s labelled orders", f"{len(frame):,}")
    return frame, batch


def find_trainer(module_name: str):
    """The member's train() function, or None if their script is not in the repo yet."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise
    if not callable(getattr(module, "train", None)):
        raise AttributeError(f"{module_name} has no train(frame, batch) function")
    return module.train


def retrain(frame: pd.DataFrame, batch: str, members=("A", "B", "C"), plots: bool = True) -> RetrainResult:
    """Run each member's training on the same frame; keep going if one member fails."""
    result = RetrainResult(batch=batch, rows=len(frame))
    t0 = time.perf_counter()
    for letter in members:
        name, module_name = TRAINERS[letter]
        try:
            train = find_trainer(module_name)
            if train is None:
                logger.warning("Skipping %s: %s.py is not in the repo yet", name, module_name.replace(".", "/"))
                result.skipped.append(name)
                continue
            logger.info("Training %s's models ...", name)
            start = time.perf_counter()
            for model, metrics in train(frame, batch=batch, plots=plots).items():
                result.runs[f"{name}/{model}"] = metrics
            logger.info("%s done in %.1fs", name, time.perf_counter() - start)
        except Exception as exc:  # one member's bug must not stop the others
            logger.exception("%s's training failed", name)
            result.failed[name] = f"{type(exc).__name__}: {exc}"
    result.seconds = time.perf_counter() - t0
    return result


def summary(result: RetrainResult) -> str:
    """Plain-text table of the runs for the end of the log."""
    lines = [f"Batch '{result.batch}': {result.rows:,} labelled orders, training took {result.seconds:.0f}s"]
    if result.runs:
        lines.append(f"{'run':<28}{'PR-AUC':>8}{'precision':>11}{'recall':>8}{'F1':>7}")
        for run, m in sorted(result.runs.items(), key=lambda kv: -kv[1].get("test_pr_auc", 0)):
            lines.append(f"{run:<28}{m.get('test_pr_auc', float('nan')):>8.3f}{m.get('test_precision', float('nan')):>11.3f}"
                         f"{m.get('test_recall', float('nan')):>8.3f}{m.get('test_f1', float('nan')):>7.3f}")
    lines += [f"SKIPPED {name}: no training script yet" for name in result.skipped]
    lines += [f"FAILED  {name}: {error}" for name, error in result.failed.items()]
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Retrain all members' models on RDS data and log them to MLflow.")
    ap.add_argument("--batch", help="Batch name for the run names (default: latest successful ETL batch)")
    ap.add_argument("--members", nargs="+", choices=sorted(TRAINERS), default=sorted(TRAINERS),
                    help="Which members' models to train (default: all)")
    ap.add_argument("--no-plots", action="store_true", help="Skip the confusion matrix and PR curve images")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        connect_mlflow()  # fail before the slow part if MLflow isn't configured
        frame, batch = load_training_frame(args.batch)
    except (RuntimeError, pymysql.MySQLError) as exc:
        logger.error("Retraining stopped: %s", exc)
        return 1
    result = retrain(frame, batch, members=args.members, plots=not args.no_plots)
    logger.info("Summary\n%s", summary(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
