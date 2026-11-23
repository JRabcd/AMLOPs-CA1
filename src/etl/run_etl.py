"""Run the ETL for one batch: read the three CSVs, clean them, load them into RDS (PBI-09).

    # new batch published as <prefix>/customers_XXX.csv, orders_XXX.csv, labels_XXX.csv
    python -m src.etl.run_etl --source https://example.com/data --batch XXX

    # or give the three files explicitly (URLs or local paths)
    python -m src.etl.run_etl --customers URL --orders URL --labels URL

Options: --force reloads a batch that already loaded successfully; --dry-run reads and cleans
but writes nothing to the database.

Every run is recorded in the etl_batches table (status, row counts, error message), and the
log shows rows read, cleaned and loaded per table (PBI-09.5). The database stores cleaned data
(docs/rds-schema.md); feature transformation (src/transform.py) happens when models are
retrained from RDS.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field

from src.cleaning import clean_batch
from src.db.config import connect, get_settings
from src.etl.extract import batch_id_for, batch_locations, extract_batch
from src.etl.load import LOAD_ORDER, batch_status, finish_batch, load_table, start_batch

logger = logging.getLogger("etl")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [etl]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@dataclass
class ETLResult:
    batch_id: str
    status: str                                   # success | skipped | dry-run | failed
    rows_read: dict[str, int] = field(default_factory=dict)
    rows_cleaned: dict[str, int] = field(default_factory=dict)
    rows_loaded: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    error: str | None = None


def run_etl(locations: dict[str, str], batch_id: str | None = None, force: bool = False,
            dry_run: bool = False) -> ETLResult:
    """Run extract -> clean -> load for one batch and return what happened."""
    t0 = time.perf_counter()
    batch_id = batch_id or batch_id_for(locations)
    result = ETLResult(batch_id=batch_id, status="failed")
    logger.info(f"=== ETL batch '{batch_id}' started{' (dry run)' if dry_run else ''} ===")

    conn = None
    if not dry_run:
        get_settings()  # fail fast on missing DB settings, before any download
        conn = connect()
        if batch_status(conn, batch_id) == "success" and not force:
            logger.info(f"Batch '{batch_id}' already loaded successfully; skipping (use --force to reload)")
            conn.close()
            result.status = "skipped"
            result.seconds = time.perf_counter() - t0
            return result
        start_batch(conn, batch_id, locations)

    try:
        raw = extract_batch(locations)
        result.rows_read = {t: len(raw[t]) for t in LOAD_ORDER}

        cleaned = clean_batch(raw["customers"], raw["orders"], raw["labels"])
        frames = {"customers": cleaned.customers, "orders": cleaned.orders, "labels": cleaned.labels}
        result.rows_cleaned = {t: len(frames[t]) for t in LOAD_ORDER}
        for t in LOAD_ORDER:
            dropped = result.rows_read[t] - result.rows_cleaned[t]
            logger.info(f"Cleaned {t}: {result.rows_read[t]:,} read -> {result.rows_cleaned[t]:,} kept ({dropped:,} dropped)")
        jc = cleaned.report.get("join_coverage", {})
        logger.info(f"Orphan labels: {jc.get('labels_to_orders_orphan', '?')} without an order, "
                    f"{jc.get('labels_to_customers_orphan', '?')} without a customer (loaded; dropped at training time)")

        if dry_run:
            result.status = "dry-run"
        else:
            for t in LOAD_ORDER:
                result.rows_loaded[t] = load_table(conn, t, frames[t], batch_id)
            finish_batch(conn, batch_id, "success", rows=result.rows_cleaned)
            result.status = "success"
    except Exception as exc:  # record any failure in etl_batches, then re-raise
        result.error = f"{type(exc).__name__}: {exc}"
        logger.error(f"Batch '{batch_id}' failed: {result.error}")
        exc.etl_logged = True  # so main() doesn't log it a second time
        if conn is not None:
            try:
                conn.rollback()
                finish_batch(conn, batch_id, "failed", error=result.error)
            except Exception as log_exc:  # don't hide the original error
                logger.error(f"Could not record the failure in etl_batches: {log_exc}")
        raise
    finally:
        if conn is not None:
            conn.close()
        result.seconds = time.perf_counter() - t0

    summary = " | ".join(f"{t}: read {result.rows_read[t]:,}, cleaned {result.rows_cleaned[t]:,}"
                         + (f", loaded {result.rows_loaded[t]:,}" if t in result.rows_loaded else "")
                         for t in LOAD_ORDER)
    logger.info(f"=== ETL batch '{batch_id}' {result.status} in {result.seconds:.1f}s === {summary}")
    return result


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load one batch of customers/orders/labels CSVs into RDS.")
    p.add_argument("--source", help="Folder or URL prefix holding <table>_<batch>.csv files")
    p.add_argument("--batch", help="Batch postfix, e.g. XXX for customers_XXX.csv (used with --source)")
    p.add_argument("--customers", help="customers CSV URL or path")
    p.add_argument("--orders", help="orders CSV URL or path")
    p.add_argument("--labels", help="labels CSV URL or path")
    p.add_argument("--batch-id", help="Override the batch name recorded in RDS")
    p.add_argument("--force", action="store_true", help="Reload a batch that already loaded successfully")
    p.add_argument("--dry-run", action="store_true", help="Read and clean only; don't write to RDS")
    args = p.parse_args(argv)
    explicit = [args.customers, args.orders, args.labels]
    if args.source or args.batch:
        if not (args.source and args.batch) or any(explicit):
            p.error("use --source with --batch, or give --customers, --orders and --labels")
    elif not all(explicit):
        p.error("give --source and --batch, or all three of --customers, --orders and --labels")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    locations = (batch_locations(args.source, args.batch) if args.source
                 else {"customers": args.customers, "orders": args.orders, "labels": args.labels})
    try:
        result = run_etl(locations, batch_id=args.batch_id or args.batch, force=args.force, dry_run=args.dry_run)
    except Exception as exc:  # e.g. missing settings or RDS unreachable before the run started
        if not getattr(exc, "etl_logged", False):
            logger.error(f"ETL stopped: {type(exc).__name__}: {exc}")
        return 1
    return 0 if result.status in ("success", "skipped", "dry-run") else 1


if __name__ == "__main__":
    sys.exit(main())
