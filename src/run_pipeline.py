"""One command for new data: load the batch into RDS, then retrain every member's models (PBI-12).

    python -m src.run_pipeline --source https://example.com/data --batch XXX
    python -m src.run_pipeline --customers URL --orders URL --labels URL

Steps (each one's time is in the summary):
  1. check   - database and MLflow settings, before anything slow
  2. ETL     - download, clean and load the three CSVs into RDS (src/etl/run_etl.py)
  3. read    - all data from RDS, old and new; only changed rows are downloaded (src/rds_cache.py)
  4. train   - every member's models at the same time, logged to MLflow as etl_<batch>_<member>_<model>

If the ETL fails, nothing is retrained. If one member's training fails, the others still run.
Exit code 0 only if every step succeeded. Must finish in 3 minutes for the CA1 demo.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from src.db.config import get_settings
from src.etl.run_etl import add_batch_args, check_batch_args, locations_from, run_etl
from src.retrain import TRAINERS, load_training_frame, retrain, summary
from src.train_common import connect_mlflow

LIMIT_SECONDS = 180

logger = logging.getLogger("pipeline")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [pipeline]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load a new batch into RDS, then retrain all members' models.")
    add_batch_args(p)
    p.add_argument("--members", nargs="+", choices=sorted(TRAINERS), default=sorted(TRAINERS),
                   help="Which members' models to train (default: all)")
    p.add_argument("--no-plots", action="store_true", help="Skip the confusion matrix and PR curve images")
    p.add_argument("--sequential", action="store_true", help="Train members one after another")
    p.add_argument("--no-cache", action="store_true", help="Download every row from RDS, not only changed rows")
    args = p.parse_args(argv)
    check_batch_args(p, args)
    return args


def run(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    steps: dict[str, float] = {}

    def mark(step: str, since: float) -> float:
        now = time.perf_counter()
        steps[step] = now - since
        return now

    t = t0
    try:  # 1. settings, so a typo fails in seconds rather than after the download
        get_settings()
        connect_mlflow()
    except Exception as exc:
        logger.error("Stopped before starting: %s: %s", type(exc).__name__, exc)
        return 1
    t = mark("check", t)

    try:  # 2. ETL
        etl = run_etl(locations_from(args), batch_id=args.batch_id or args.batch, force=args.force)
    except Exception as exc:
        if not getattr(exc, "etl_logged", False):
            logger.error("ETL failed: %s: %s", type(exc).__name__, exc)
        logger.error("Nothing was retrained because the ETL failed.")
        return 1
    if etl.status == "skipped":
        logger.warning("Batch '%s' was already in RDS; retraining on the data that is there.", etl.batch_id)
    t = mark("ETL", t)

    try:  # 3. read old + new data from RDS
        frame, batch = load_training_frame(etl.batch_id, use_cache=not args.no_cache)
    except Exception as exc:
        logger.error("Could not read the training data from RDS: %s: %s", type(exc).__name__, exc)
        return 1
    t = mark("read RDS", t)

    # 4. retrain everyone
    result = retrain(frame, batch, members=args.members, plots=not args.no_plots, parallel=not args.sequential)
    mark("train + log", t)

    total = time.perf_counter() - t0
    timing = " | ".join(f"{k} {v:.0f}s" for k, v in steps.items())
    verdict = "within" if total <= LIMIT_SECONDS else "OVER"
    rows = ", ".join(f"{k} {v:,}" for k, v in (etl.rows_loaded or etl.rows_cleaned).items())
    logger.info("Pipeline summary\nETL batch '%s': %s (%s)\n%s\nTime: %s | total %.0fs (%s the %ds limit)",
                etl.batch_id, etl.status, rows or "no rows loaded", summary(result), timing, total, verdict, LIMIT_SECONDS)
    return 0 if result.ok else 1


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
