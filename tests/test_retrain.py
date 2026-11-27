"""Unit tests for retraining (PBI-10). No database or MLflow server needed."""
import sys
import types

import numpy as np
import pandas as pd
import pytest

from src import retrain as rt
from src import train_common as tc
from src.cleaning import clean_batch

CUSTOMERS_CSV = {
    "country_code": ["my", "PK"], "customer_id": ["c1", "c2"], "mobile_verified": ["yes", "False"],
    "num_orders_last_50days": ["3", "0"], "num_cancelled_orders_last_50days": ["0", "0"],
    "num_refund_orders_last_50days": ["1", "0"], "total_payment_last_50days": ["42.5", "0"],
    "num_associated_customers": ["1", "1"], "first_order_datetime": ["2021-03-23 09:16:29", ""],
}
ORDERS_CSV = {
    "country_code": ["MY", "PK"], "order_id": ["o1", "o1"], "collect_type": ["Delivery", "delivery"],
    "payment_method": ["credit card", ""], "order_value": ["12.5", "3"], "num_items_ordered": ["2", "1"],
    "refund_value": ["0.0", "0.0"], "order_date": ["2023/04/09", "2023/Feb/11"],
}
LABELS_CSV = {"country_code": ["my", "PK"], "order_id": ["o1", "o1"], "customer_id": ["c1", "c2"], "is_fraud": ["0", "1"]}

# The same two customers/orders/labels as pymysql returns them from RDS (with retrain's converters).
RDS_ROWS = {
    "customers": [("c1", "MY", 1, 3, 0, 1, 42.5, 1, "2021-03-23 09:16:29"), ("c2", "PK", 0, 0, 0, 0, 0.0, 1, None)],
    "orders": [("MY", "o1", "delivery", "credit card", 12.5, 2, 0.0, "2023-04-09"),
               ("PK", "o1", "delivery", None, 3.0, 1, 0.0, "2023-02-11")],
    "labels": [("MY", "o1", "c1", 0), ("PK", "o1", "c2", 1)],
}


class FakeConn:
    """Answers SELECTs on the three tables and etl_batches like pymysql does."""

    def __init__(self, batches=(("base",),)):
        self.batches, self.result, self.closed = list(batches), [], False

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        assert sql.startswith("SELECT")  # retraining only reads
        if sql == "SELECT NOW()":
            self.result = [("2026-11-26 12:00:00",)]
            return
        table = sql.split(" FROM ")[1].split()[0]
        rows = self.batches if table == "etl_batches" else RDS_ROWS[table]
        self.result = [(len(rows),)] if "COUNT(*)" in sql else rows

    def fetchall(self):
        return self.result

    def fetchone(self):
        return self.result[0] if self.result else None

    def close(self):
        self.closed = True


def test_rds_tables_get_the_same_dtypes_and_values_as_cleaning():
    cleaned = clean_batch(*(pd.DataFrame(d) for d in (CUSTOMERS_CSV, ORDERS_CSV, LABELS_CSV)))
    for table in ("customers", "orders", "labels"):
        from_rds = rt.read_table(FakeConn(), table)
        expected = getattr(cleaned, table)[from_rds.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(from_rds, expected, check_dtype=True)


def test_latest_batch():
    assert rt.latest_batch(FakeConn(batches=[("B7",)])) == "B7"
    with pytest.raises(RuntimeError, match="run_etl"):
        rt.latest_batch(FakeConn(batches=[]))


@pytest.mark.parametrize("use_cache", [False, True])
def test_load_training_frame_joins_rds_tables(monkeypatch, tmp_path, use_cache):
    conn = FakeConn()
    monkeypatch.setattr(rt, "connect", lambda **kw: conn)
    monkeypatch.setattr(rt, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("src.rds_cache._identity", lambda: "fake-db")
    for _ in range(2):  # with the cache, the second read comes from the local copy
        frame, batch = rt.load_training_frame(use_cache=use_cache)
        assert batch == "base" and len(frame) == 2 and conn.closed
        assert frame.set_index("country_code").is_fraud.to_dict() == {"MY": 0, "PK": 1}
    assert (tmp_path / "cache").exists() == use_cache
    assert rt.load_training_frame("X1", use_cache=False)[1] == "X1"


def fake_trainer(monkeypatch, name, train):
    module = types.ModuleType(name)
    if train is not None:
        module.train = train
    monkeypatch.setitem(sys.modules, name, module)


def test_retrain_skips_missing_scripts_and_survives_a_failure(monkeypatch):
    monkeypatch.setattr(rt, "TRAINERS", {"A": ("Ann", "fake_a"), "B": ("Ben", "fake_b"), "C": ("Cat", "fake_c_missing")})
    calls = []
    fake_trainer(monkeypatch, "fake_a", lambda frame, batch, plots: calls.append(batch) or {"m1": {"test_pr_auc": 0.5}})
    fake_trainer(monkeypatch, "fake_b", lambda frame, batch, plots: 1 / 0)
    result = rt.retrain(pd.DataFrame({"x": [1]}), "B9")
    assert calls == ["B9"] and set(result.runs) == {"Ann/m1"}
    assert result.skipped == ["Cat"] and "ZeroDivisionError" in result.failed["Ben"]
    assert not result.ok
    text = rt.summary(result)
    assert "Ann/m1" in text and "SKIPPED Cat" in text and "FAILED  Ben" in text


def test_retrain_ok_only_when_every_member_that_ran_succeeded(monkeypatch):
    monkeypatch.setattr(rt, "TRAINERS", {"A": ("Ann", "fake_a"), "C": ("Cat", "fake_c_missing")})
    fake_trainer(monkeypatch, "fake_a", lambda frame, batch, plots: {"m1": {"test_pr_auc": 0.5}})
    assert rt.retrain(pd.DataFrame(), "B1", members=["A", "C"]).ok
    assert not rt.retrain(pd.DataFrame(), "B1", members=["C"]).ok  # nothing trained at all


def test_script_without_train_function_is_an_error(monkeypatch):
    fake_trainer(monkeypatch, "fake_no_train", None)
    with pytest.raises(AttributeError, match="train"):
        rt.find_trainer("fake_no_train")
    assert rt.find_trainer("src.train_memberZ") is None


def test_main_stops_early_without_mlflow(monkeypatch):
    monkeypatch.setattr(tc, "load_dotenv", lambda: None)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(rt, "load_training_frame", lambda batch: pytest.fail("must not read RDS"))
    assert rt.main([]) == 1


def test_member_a_runs_are_named_by_batch(monkeypatch, tmp_path):
    """End to end with a local MLflow store: member A's two runs land under the batch name."""
    rng = np.random.default_rng(0)
    n = 800
    amount = rng.gamma(2, 10, n)
    frame = pd.DataFrame({
        "country_code": rng.choice(["MY", "PK"], n), "order_id": [f"o{i}" for i in range(n)],
        "customer_id": [f"c{i}" for i in rng.integers(0, 200, n)], "payment_method": "balance",
        "collect_type": "delivery", "refund_value": 0.0, "order_value": amount,
        "order_date": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 100, n), unit="D"),
        "num_orders_last_50days": rng.integers(0, 20, n),
        "is_fraud": (rng.random(n) < np.clip(amount / 100, 0.02, 0.9)).astype(int),
    })
    monkeypatch.setattr(tc, "load_dotenv", lambda: None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    monkeypatch.chdir(tmp_path)
    tc.connect_mlflow()
    result = rt.retrain(frame, "T2", members=["A"], plots=False)
    assert result.ok and set(result.runs) == {"QianYunyang/lgbm_v1", "QianYunyang/lgbm_v2"}
    names = set(tc.mlflow.search_runs(experiment_names=[tc.EXPERIMENT])["tags.mlflow.runName"])
    assert names == {"etl_T2_QianYunyang_lgbm_v1", "etl_T2_QianYunyang_lgbm_v2"}
