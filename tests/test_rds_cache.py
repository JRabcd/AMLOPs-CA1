"""Unit tests for the local copy of RDS tables (PBI-12). No database needed."""
import pandas as pd
import pytest

from src import rds_cache
from src.retrain import to_clean_types


class FakeDB:
    """A tiny in-memory 'labels' table with loaded_at, queried the way rds_cache queries RDS."""

    def __init__(self):
        self.rows = {}   # (country, order) -> [country, order, customer, is_fraud, loaded_at]
        self.clock = 0
        self.fetched = []   # rows sent back by each data query, to see what was downloaded

    def upsert(self, country, order, customer, fraud):
        self.clock += 1
        self.rows[(country, order)] = [country, order, customer, fraud, self.clock]

    def delete_all(self):
        self.rows.clear()

    # pymysql-like connection/cursor
    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        assert sql.startswith("SELECT")
        if sql == "SELECT NOW()":
            self.result = [(self.clock,)]
        elif "COUNT(*)" in sql:
            self.result = [(len(self.rows),)]
        else:
            since = int(params[0]) if params else None
            self.result = [tuple(r[:4]) for r in self.rows.values() if since is None or r[4] >= since]
            self.fetched.append(len(self.result))

    def fetchone(self):
        return self.result[0]

    def fetchall(self):
        return self.result


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setattr(rds_cache, "_identity", lambda: "fake-db")
    d = FakeDB()
    for i in range(5):
        d.upsert("MY", f"o{i}", f"c{i}", 0)
    return d


def read(db, cache_dir):
    return rds_cache.read_tables(db, to_clean_types, tables=("labels",), cache_dir=cache_dir)["labels"]


def as_set(df):
    return set(map(tuple, df.astype(object).to_numpy().tolist()))


def test_first_read_downloads_everything_then_only_changes(db, tmp_path):
    first = read(db, tmp_path)
    assert len(first) == 5 and db.fetched == [5]
    db.upsert("PK", "o9", "c9", 1)           # new row from a new batch
    db.upsert("MY", "o1", "c1", 1)           # existing row changed by the new batch
    second = read(db, tmp_path)
    assert db.fetched[-1] <= 3               # only changed rows (plus any in the same second)
    full = read(db, None)
    assert as_set(second) == as_set(full) and len(second) == 6
    assert second.set_index("order_id").loc["o1", "is_fraud"] == 1


def test_cached_dtypes_match_a_fresh_read(db, tmp_path):
    read(db, tmp_path)
    cached = read(db, tmp_path)
    pd.testing.assert_series_equal(cached.dtypes, read(db, None).dtypes)


def test_mismatch_with_rds_triggers_a_full_read(db, tmp_path):
    read(db, tmp_path)
    db.delete_all()                          # e.g. the database was recreated
    db.upsert("TH", "x1", "c1", 0)
    out = read(db, tmp_path)
    assert list(out.order_id) == ["x1"] and db.fetched[-1] == 1


def test_copy_from_another_database_is_ignored(db, tmp_path, monkeypatch):
    read(db, tmp_path)
    monkeypatch.setattr(rds_cache, "_identity", lambda: "other-db")
    read(db, tmp_path)
    assert db.fetched[-1] == 5               # full read, not a delta


def test_broken_copy_is_ignored(db, tmp_path):
    read(db, tmp_path)
    (tmp_path / "labels.parquet").write_text("not parquet")
    assert len(read(db, tmp_path)) == 5 and db.fetched[-1] == 5


def test_no_cache_dir_keeps_no_copy(db, tmp_path):
    read(db, None)
    assert not any(tmp_path.iterdir())
