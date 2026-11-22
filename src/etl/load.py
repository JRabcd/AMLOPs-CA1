"""Load step of the ETL (PBI-09.3): write cleaned tables to RDS without duplicate keys.

Every table is written with INSERT ... ON DUPLICATE KEY UPDATE, so re-running a batch, or a later
batch that repeats a customer or order, updates the existing row instead of adding a duplicate.
Columns and keys follow sql/schema.sql and docs/rds-schema.md.
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger("etl")

# Data columns per table, in schema order; batch_id is added on load.
COLUMNS: dict[str, list[str]] = {
    "customers": [
        "customer_id", "country_code", "mobile_verified", "num_orders_last_50days",
        "num_cancelled_orders_last_50days", "num_refund_orders_last_50days",
        "total_payment_last_50days", "num_associated_customers", "first_order_datetime",
    ],
    "orders": [
        "country_code", "order_id", "collect_type", "payment_method", "order_value",
        "num_items_ordered", "refund_value", "order_date",
    ],
    "labels": ["country_code", "order_id", "customer_id", "is_fraud"],
}
KEYS: dict[str, list[str]] = {
    "customers": ["customer_id"],
    "orders": ["country_code", "order_id"],
    "labels": ["country_code", "order_id"],
}
DATE_ONLY = {"order_date"}
LOAD_ORDER = ("customers", "orders", "labels")


def upsert_sql(table: str) -> str:
    """INSERT ... ON DUPLICATE KEY UPDATE for one table (all non-key columns are updated)."""
    cols = COLUMNS[table] + ["batch_id"]
    updates = [c for c in cols if c not in KEYS[table]]
    return (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
        f"ON DUPLICATE KEY UPDATE {', '.join(f'{c}=VALUES({c})' for c in updates)}"
    )


def _py(value, date_only: bool):
    """Convert one pandas/numpy value to what PyMySQL expects (missing -> None)."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.date() if date_only else value
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    return value


def to_rows(df: pd.DataFrame, table: str, batch_id: str) -> list[tuple]:
    """Turn a cleaned DataFrame into row tuples in schema order, ending with batch_id."""
    cols = COLUMNS[table]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Cleaned {table} is missing column(s): {missing}")
    if df[KEYS[table]].isna().any().any():
        raise ValueError(f"Cleaned {table} has rows with a missing key ({KEYS[table]})")
    flags = [c in DATE_ONLY for c in cols]
    data = df[cols].astype(object).to_numpy()
    return [tuple(_py(v, f) for v, f in zip(row, flags)) + (batch_id,) for row in data]


def load_table(conn, table: str, df: pd.DataFrame, batch_id: str, chunk_size: int = 5000) -> int:
    """Upsert one cleaned table in chunks and commit. Returns the number of rows sent."""
    rows = to_rows(df, table, batch_id)
    sql = upsert_sql(table)
    with conn.cursor() as cur:
        for start in range(0, len(rows), chunk_size):
            cur.executemany(sql, rows[start:start + chunk_size])
    conn.commit()
    logger.info(f"Loaded {table}: {len(rows):,} rows upserted")
    return len(rows)


# ---- etl_batches: one row per ETL run (see docs/rds-schema.md) -------------------------------

def batch_status(conn, batch_id: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM etl_batches WHERE batch_id = %s", (batch_id,))
        row = cur.fetchone()
    return row[0] if row else None


def start_batch(conn, batch_id: str, locations: dict[str, str]) -> None:
    """Record (or reset, for a retry) a batch as running."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO etl_batches (batch_id, customers_url, orders_url, labels_url, status, started_at) "
            "VALUES (%s, %s, %s, %s, 'running', NOW()) "
            "ON DUPLICATE KEY UPDATE customers_url=VALUES(customers_url), orders_url=VALUES(orders_url), "
            "labels_url=VALUES(labels_url), status='running', started_at=NOW(), finished_at=NULL, "
            "error_message=NULL, rows_customers=NULL, rows_orders=NULL, rows_labels=NULL",
            (batch_id, *(str(locations.get(t))[:500] for t in LOAD_ORDER)),
        )
    conn.commit()


def finish_batch(conn, batch_id: str, status: str, rows: dict[str, int] | None = None, error: str | None = None) -> None:
    """Mark a batch success or failed, with the cleaned row counts or the error."""
    rows = rows or {}
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE etl_batches SET status=%s, finished_at=NOW(), rows_customers=%s, rows_orders=%s, "
            "rows_labels=%s, error_message=%s WHERE batch_id=%s",
            (status, rows.get("customers"), rows.get("orders"), rows.get("labels"),
             (error or None) and error[:500], batch_id),
        )
    conn.commit()
