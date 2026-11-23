"""Unit tests for the ETL (PBI-09). No database or network needed."""
import io
import urllib.error

import numpy as np
import pandas as pd
import pytest

from src.db import config
from src.etl import extract, load
from src.etl.run_etl import parse_args, run_etl

CUSTOMERS_CSV = (
    "country_code,customer_id,mobile_verified,num_orders_last_50days,num_cancelled_orders_last_50days,"
    "num_refund_orders_last_50days,total_payment_last_50days,num_associated_customers,first_order_datetime\n"
    "my,c1,yes,3,0,1,42.5,1,2021-03-23 09:16:29\n"
    "PK,c2,False,0,0,0,0,1,\n"
)
ORDERS_CSV = (
    "country_code,order_id,collect_type,payment_method,order_value,num_items_ordered,refund_value,order_date\n"
    "MY,o1,Delivery,credit card,12.5,2,0.0,2023/04/09\n"
    "PK,o1,delivery,,0,1,0.0,2023/Feb/11\n"
)
LABELS_CSV = "country_code,order_id,customer_id,is_fraud\nmy,o1,c1,0\nPK,o1,c2,1\n"


@pytest.fixture
def batch_dir(tmp_path):
    (tmp_path / "customers_T1.csv").write_text(CUSTOMERS_CSV)
    (tmp_path / "orders_T1.csv").write_text(ORDERS_CSV)
    (tmp_path / "labels_T1.csv").write_text(LABELS_CSV)
    return tmp_path


# ---- extract (PBI-09.1) ------------------------------------------------------------------

def test_batch_id_from_url_and_path():
    assert extract.batch_id_from_name("https://x.com/data/orders_XXX.csv?raw=1") == "XXX"
    assert extract.batch_id_from_name(r"C:\data\labels_base.csv") == "base"
    with pytest.raises(extract.ExtractError):
        extract.batch_id_from_name("orders.csv")


def test_batch_locations_and_mismatched_batches():
    locs = extract.batch_locations("https://x.com/data", "B2")
    assert locs["customers"] == "https://x.com/data/customers_B2.csv"
    assert extract.batch_id_for(locs) == "B2"
    with pytest.raises(extract.ExtractError, match="different batches"):
        extract.batch_id_for({**locs, "labels": "https://x.com/data/labels_B3.csv"})


def test_read_table_missing_file(tmp_path):
    with pytest.raises(extract.ExtractError, match="File not found"):
        extract.read_table(str(tmp_path / "orders_X.csv"), "orders")


def test_read_table_wrong_columns(tmp_path):
    bad = tmp_path / "orders_X.csv"
    bad.write_text("country_code,order_id\nMY,o1\n")
    with pytest.raises(extract.ExtractError, match="missing column"):
        extract.read_table(str(bad), "orders")


def test_read_table_reads_everything_as_text(batch_dir):
    df = extract.read_table(str(batch_dir / "orders_T1.csv"), "orders")
    assert len(df) == 2 and df["order_value"].iloc[0] == "12.5"


def test_download_404_fails_fast(monkeypatch):
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        raise urllib.error.HTTPError(url, 404, "Not Found", None, io.BytesIO())

    monkeypatch.setattr(extract.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(extract.ExtractError, match="HTTP 404"):
        extract.read_table("https://x.com/orders_X.csv", "orders")
    assert len(calls) == 1  # no pointless retries on 4xx


def test_download_retries_then_gives_up(monkeypatch):
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(extract.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(extract.ExtractError, match="after 3 attempts"):
        extract.read_table("https://x.com/orders_X.csv", "orders", backoff=0)
    assert len(calls) == 3


# ---- load (PBI-09.3) ---------------------------------------------------------------------

def test_upsert_sql_updates_only_non_key_columns():
    sql = load.upsert_sql("orders")
    assert sql.startswith("INSERT INTO orders (country_code, order_id,")
    assert "ON DUPLICATE KEY UPDATE" in sql
    update_part = sql.split("ON DUPLICATE KEY UPDATE", 1)[1]
    assert "order_id=" not in update_part and "country_code=" not in update_part
    assert "batch_id=VALUES(batch_id)" in update_part


def test_to_rows_converts_missing_bool_int_and_dates():
    df = pd.DataFrame({
        "country_code": pd.array(["MY"], dtype="string"),
        "order_id": pd.array(["o1"], dtype="string"),
        "collect_type": pd.array(["delivery"], dtype="string"),
        "payment_method": pd.array([pd.NA], dtype="string"),
        "order_value": [np.nan],
        "num_items_ordered": pd.array([3], dtype="Int64"),
        "refund_value": [0.0],
        "order_date": pd.to_datetime(["2023-04-09"]),
    })
    (row,) = load.to_rows(df, "orders", "B1")
    assert row[3] is None and row[4] is None              # NULL convention
    assert row[5] == 3 and type(row[5]) is int
    assert str(row[7]) == "2023-04-09" and not hasattr(row[7], "hour")  # DATE, not DATETIME
    assert row[-1] == "B1"


def test_to_rows_booleans_and_missing_datetime():
    df = pd.DataFrame({c: [None] for c in load.COLUMNS["customers"]})
    df["customer_id"] = ["c1"]
    df["country_code"] = ["MY"]
    df["mobile_verified"] = pd.array([True], dtype="boolean")
    df["first_order_datetime"] = pd.to_datetime([None])
    (row,) = load.to_rows(df, "customers", "B1")
    assert row[2] == 1 and row[8] is None


def test_to_rows_rejects_missing_key():
    df = pd.DataFrame({c: ["x"] for c in load.COLUMNS["labels"]})
    df.loc[0, "order_id"] = None
    with pytest.raises(ValueError, match="missing key"):
        load.to_rows(df, "labels", "B1")


# ---- config (PBI-09.4) -------------------------------------------------------------------

def test_config_requires_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)  # no .env here
    for k in ("DB_HOST", "DB_USER", "DB_PASSWORD", "DB_PORT", "DB_NAME"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(config.ConfigError, match="DB_HOST"):
        config.get_settings()


def test_config_reads_env_and_hides_password(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("DB_HOST", "h"); monkeypatch.setenv("DB_USER", "u")
    monkeypatch.setenv("DB_PASSWORD", "secret"); monkeypatch.setenv("DB_PORT", "3307")
    s = config.get_settings()
    assert (s.host, s.port, s.database) == ("h", 3307, "spbuy")
    assert "secret" not in repr(s)


# ---- run (PBI-09.2 / 09.5) ---------------------------------------------------------------

def test_dry_run_reads_and_cleans_without_db(batch_dir, monkeypatch):
    monkeypatch.setattr("src.etl.run_etl.connect", lambda *a, **k: pytest.fail("dry run must not connect"))
    result = run_etl(extract.batch_locations(str(batch_dir), "T1"), dry_run=True)
    assert result.status == "dry-run" and result.batch_id == "T1"
    assert result.rows_read == {"customers": 2, "orders": 2, "labels": 2}
    assert result.rows_cleaned["orders"] == 2  # o1 in MY and PK are different orders


def test_cli_argument_rules():
    assert parse_args(["--source", "s", "--batch", "B"]).batch == "B"
    with pytest.raises(SystemExit):
        parse_args(["--source", "s"])
    with pytest.raises(SystemExit):
        parse_args(["--customers", "a", "--orders", "b"])
