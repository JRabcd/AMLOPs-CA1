"""Local copy of the RDS tables, so retraining downloads only what changed (PBI-12).

Reading all ~2.8 million rows from RDS takes about 25 s on a fast network but several minutes on a
slow one, which alone breaks the 3-minute demo limit. Every row has a loaded_at timestamp that MySQL
updates whenever the ETL inserts or changes it, so after one full read we only fetch rows with
loaded_at at or after the previous read, and merge them in by primary key.

Safety checks: the copy is only used for the same database host and schema, and if a table's row
count in RDS doesn't match the merged copy (e.g. the database was recreated), that table is
downloaded again in full. Rows are never deleted by the ETL, so the count check is enough.

The copy lives in .cache/rds/ (git-ignored). Delete that folder, or pass --no-cache, to force a
full read.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.db.config import REPO_ROOT, get_settings
from src.etl.load import COLUMNS, KEYS

logger = logging.getLogger("retrain")
CACHE_DIR = REPO_ROOT / ".cache" / "rds"


def server_now(conn) -> str:
    """The database's current time, as text; used as the next read's starting point."""
    with conn.cursor() as cur:
        cur.execute("SELECT NOW()")
        return str(cur.fetchone()[0])


def count_rows(conn, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        return int(cur.fetchone()[0])


def fetch(conn, table: str, since: str | None, to_frame) -> pd.DataFrame:
    """Rows of one table (all of them, or those loaded at or after `since`) as a DataFrame."""
    cols = COLUMNS[table]
    sql = f"SELECT {', '.join(cols)} FROM {table}"
    with conn.cursor() as cur:
        if since is None:
            cur.execute(sql)
        else:
            cur.execute(sql + " WHERE loaded_at >= %s", (since,))
        rows = cur.fetchall()
    return to_frame(pd.DataFrame(rows, columns=cols))


def _identity() -> str:
    s = get_settings()
    return f"{s.host}:{s.port}/{s.database}"


def _load(cache_dir: Path, table: str, identity: str) -> tuple[pd.DataFrame, str] | None:
    meta_path, data_path = cache_dir / f"{table}.json", cache_dir / f"{table}.parquet"
    if not (meta_path.exists() and data_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("database") != identity:
            return None
        return pd.read_parquet(data_path), meta["read_at"]
    except Exception as exc:  # a broken copy just means a full read
        logger.warning("Ignoring unreadable local copy of %s: %s", table, exc)
        return None


def _save(cache_dir: Path, table: str, identity: str, frame: pd.DataFrame, read_at: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f"{table}.parquet.tmp"
    frame.to_parquet(tmp, index=False)
    tmp.replace(cache_dir / f"{table}.parquet")
    (cache_dir / f"{table}.json").write_text(json.dumps({"database": identity, "read_at": read_at}), encoding="utf-8")


def read_tables(conn, to_frame, tables=("customers", "orders", "labels"), cache_dir: Path | None = CACHE_DIR,
                ) -> dict[str, pd.DataFrame]:
    """All rows of each table, downloading only changed rows when a valid local copy exists.

    to_frame converts the raw rows to the cleaned dtypes (retrain.to_clean_types).
    cache_dir=None always reads everything and keeps no copy.
    """
    read_at = server_now(conn)  # before reading, so rows changed during the read are fetched next time
    identity = _identity() if cache_dir is not None else ""
    out = {}
    for table in tables:
        cached = _load(cache_dir, table, identity) if cache_dir is not None else None
        unchanged = False
        if cached is not None:
            old, since = cached
            new = fetch(conn, table, since, to_frame)
            merged = pd.concat([old, new], ignore_index=True) if len(new) else old
            merged = merged.drop_duplicates(KEYS[table], keep="last").reset_index(drop=True)
            expected = count_rows(conn, table)
            if len(merged) == expected:
                logger.info("%s: %s changed rows downloaded, %s from the local copy", table, f"{len(new):,}", f"{len(old):,}")
                out[table] = merged
                unchanged = len(new) == 0
            else:
                logger.warning("%s: local copy doesn't match RDS (%s vs %s rows), downloading it all",
                               table, f"{len(merged):,}", f"{expected:,}")
                out[table] = fetch(conn, table, None, to_frame)
        else:
            out[table] = fetch(conn, table, None, to_frame)
            logger.info("%s: %s rows downloaded (no local copy yet)", table, f"{len(out[table]):,}")
        if cache_dir is not None and not unchanged:  # nothing new: keep the old copy and its start time
            _save(cache_dir, table, identity, out[table], read_at)
    return out
