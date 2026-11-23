"""Unit tests for the training code (PBI-07a). Small synthetic data, no MLflow server needed."""
import numpy as np
import pandas as pd
import pytest

from src import train_common as tc
from src import train_memberA


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    n = 1200
    amount = rng.gamma(2, 10, n)
    f = pd.DataFrame({
        "country_code": rng.choice(["MY", "PK", "PH"], n),
        "order_id": [f"o{i}" for i in range(n)],
        "customer_id": [f"c{i}" for i in rng.integers(0, 300, n)],
        "payment_method": rng.choice(["credit card", "balance"], n),
        "collect_type": "delivery",
        "order_date": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 120, n), unit="D"),
        "order_value": amount,
        "refund_value": 0.0,
        "num_orders_last_50days": rng.integers(0, 20, n),
    })
    f["is_fraud"] = (rng.random(n) < np.clip(amount / 100, 0.02, 0.9)).astype(int)
    return f


def test_split_has_no_overlap_and_tests_on_newest_orders(frame):
    tr, va, te = tc.split(frame)
    assert sorted(np.concatenate([tr, va, te])) == list(range(len(frame)))
    assert not set(frame.customer_id.iloc[tr]) & set(frame.customer_id.iloc[va])
    assert frame.order_date.iloc[np.concatenate([tr, va])].max() <= frame.order_date.iloc[te].min()
    assert len(te) == len(frame) - int(len(frame) * 0.8)


def test_split_is_repeatable(frame):
    assert all(np.array_equal(a, b) for a, b in zip(tc.split(frame), tc.split(frame)))


def test_matrices_have_matching_columns_and_no_leaky_ones(frame):
    (Xtr, ytr), (Xva, yva), (Xte, yte), _ = tc.make_matrices(frame, *tc.split(frame))
    assert list(Xtr.columns) == list(Xva.columns) == list(Xte.columns)
    assert len(Xva) == len(yva) and len(Xte) == len(yte)
    assert not {"refund_value", "is_fraud", "customer_id", "order_id"} & set(Xtr.columns)


def test_validation_and_test_rows_line_up_with_their_labels(frame):
    tr, va, te = tc.split(frame)
    (_, _), (Xva, yva), (Xte, yte), _ = tc.make_matrices(frame, tr, va, te)
    assert np.allclose(Xva["order_value"], frame.order_value.iloc[va])
    assert np.allclose(Xte["order_value"], frame.order_value.iloc[te])
    assert np.array_equal(yva, frame.is_fraud.iloc[va])


def test_drop_unusable_removes_orphans(frame):
    broken = frame.head(5).copy()
    broken.loc[broken.index[0], "order_date"] = pd.NaT
    broken["num_orders_last_50days"] = broken["num_orders_last_50days"].astype("Int64")
    broken.loc[broken.index[1], "num_orders_last_50days"] = pd.NA
    assert len(tc.drop_unusable(broken)) == 3


def test_best_threshold_separates_perfect_scores():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.7, 0.9])
    assert tc.best_threshold(y, p) == pytest.approx(0.7)


def test_score_counts_false_alarms_on_genuine_orders():
    m = tc.score(np.array([0, 0, 0, 0, 1]), np.array([0.9, 0.1, 0.1, 0.1, 0.8]), 0.5)
    assert m["false_alarms"] == 1 and m["false_alarm_rate"] == 0.25
    assert m["precision"] == 0.5 and m["recall"] == 1.0


def test_mlflow_must_be_configured(monkeypatch):
    monkeypatch.setattr(tc, "load_dotenv", lambda: None)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        tc.connect_mlflow()


def test_order_month_is_added_without_changing_the_input(frame):
    out = train_memberA.add_order_month(frame)
    assert "order_month" not in frame and out.order_month.between(1, 12).all()


def test_versions_differ_as_documented():
    assert train_memberA.V1["class_weight"] == "balanced" and "class_weight" not in train_memberA.V2


def test_both_versions_train_and_log(frame, tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "load_dotenv", lambda: None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    monkeypatch.chdir(tmp_path)
    tc.connect_mlflow()
    res = train_memberA.train(frame, batch="T1", plots=False)
    runs = tc.mlflow.search_runs(experiment_names=[tc.EXPERIMENT])
    assert set(runs["tags.mlflow.runName"]) == {"etl_T1_QianYunyang_lgbm_v1", "etl_T1_QianYunyang_lgbm_v2"}
    assert {"test_pr_auc", "test_precision", "test_recall", "test_f1", "test_false_alarm_rate"} <= set(res["lgbm_v2"])
