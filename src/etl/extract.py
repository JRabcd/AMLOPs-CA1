"""Extract step of the ETL (PBI-09.1): read a batch's three CSVs and check their columns.

A batch is three files with the same postfix, e.g. customers_XXX.csv, orders_XXX.csv and
labels_XXX.csv. They can be public URLs (the operations team publishes new data that way) or
local paths. Every value is read as text (dtype=str); typing and fixing values is the job of
src/cleaning.py.
"""
from __future__ import annotations

import io
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from src.cleaning import validate_schema

logger = logging.getLogger("etl")

TABLES = ("customers", "orders", "labels")
_NAME = re.compile(r"^(customers|orders|labels)_(?P<batch>[A-Za-z0-9][A-Za-z0-9._-]*)\.csv$", re.IGNORECASE)


class ExtractError(RuntimeError):
    """Raised when a batch file can't be read or doesn't have the expected columns."""


def is_url(location: str) -> bool:
    return location.lower().startswith(("http://", "https://"))


def batch_id_from_name(location: str) -> str:
    """customers_XXX.csv (path or URL) -> 'XXX'."""
    name = location.split("?", 1)[0].rstrip("/").split("/")[-1].split("\\")[-1]
    m = _NAME.match(name)
    if not m:
        raise ExtractError(f"Can't tell the batch from {name!r}: expected <table>_<batch>.csv, e.g. orders_XXX.csv")
    return m.group("batch")


def batch_locations(source: str, batch: str) -> dict[str, str]:
    """Build the three file locations from a folder or URL prefix and a batch postfix."""
    sep = "" if source.endswith(("/", "\\")) else "/"
    return {t: f"{source}{sep}{t}_{batch}.csv" for t in TABLES}


def batch_id_for(locations: dict[str, str]) -> str:
    """Check all three files share one postfix and return it."""
    ids = {t: batch_id_from_name(loc) for t, loc in locations.items()}
    if len(set(ids.values())) != 1:
        raise ExtractError(f"The three files are from different batches: {ids}")
    return next(iter(ids.values()))


def _download(url: str, retries: int, timeout: float, backoff: float) -> bytes:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:  # 404/403 won't fix themselves
                raise ExtractError(f"Could not download {url}: HTTP {exc.code} {exc.reason}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt < retries:
            logger.warning(f"Download failed ({last}); retry {attempt}/{retries - 1} for {url}")
            time.sleep(backoff * attempt)
    raise ExtractError(f"Could not download {url} after {retries} attempts: {last}")


def read_table(location: str, table: str, retries: int = 3, timeout: float = 60, backoff: float = 2) -> pd.DataFrame:
    """Read one batch file as text and check it has exactly the expected columns."""
    if table not in TABLES:
        raise ExtractError(f"Unknown table {table!r}")
    if is_url(location):
        raw = _download(location, retries, timeout, backoff)
        source = io.BytesIO(raw)
    else:
        path = Path(location)
        if not path.is_file():
            raise ExtractError(f"File not found for {table}: {location}")
        source = path
    try:
        df = pd.read_csv(source, dtype=str)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ExtractError(f"{table} file {location} is not a readable CSV: {exc}") from exc
    try:
        validate_schema(df, table)
    except ValueError as exc:
        raise ExtractError(f"{table} file {location}: {exc}") from exc
    return df


def extract_batch(locations: dict[str, str], **kwargs) -> dict[str, pd.DataFrame]:
    """Read and check all three files of a batch. Logs rows read per table."""
    missing = [t for t in TABLES if t not in locations]
    if missing:
        raise ExtractError(f"Missing file location(s) for: {', '.join(missing)}")
    frames = {}
    for table in TABLES:
        frames[table] = read_table(locations[table], table, **kwargs)
        logger.info(f"Read {table}: {len(frames[table]):,} rows from {locations[table]}")
    return frames
