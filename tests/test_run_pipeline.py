"""Unit tests for the one-command pipeline (PBI-12). No database, network or MLflow needed."""
import logging

import pandas as pd
import pytest

from src import run_pipeline as rp
from src.etl.run_etl import ETLResult
from src.retrain import RetrainResult


@pytest.fixture
def calls(monkeypatch):
    """Replace every slow step with a recorder; tests change what each step returns."""
    log = {"steps": []}
    monkeypatch.setattr(rp, "get_settings", lambda: log["steps"].append("settings"))
    monkeypatch.setattr(rp, "connect_mlflow", lambda: log["steps"].append("mlflow"))

    def fake_etl(locations, batch_id=None, force=False):
        log["steps"].append("etl")
        log["etl_args"] = (locations, batch_id, force)
        return log.get("etl_result", ETLResult(batch_id=batch_id, status="success",
                                               rows_loaded={"customers": 3, "orders": 4, "labels": 4}))

    def fake_read(batch, use_cache=True):
        log["steps"].append("read")
        log["read_args"] = (batch, use_cache)
        return pd.DataFrame({"x": [1]}), batch

    def fake_retrain(frame, batch, members, plots, parallel):
        log["steps"].append("train")
        log["train_args"] = (batch, members, plots, parallel)
        return log.get("train_result", RetrainResult(batch=batch, runs={"A/m": {"test_pr_auc": 0.5}}))

    monkeypatch.setattr(rp, "run_etl", fake_etl)
    monkeypatch.setattr(rp, "load_training_frame", fake_read)
    monkeypatch.setattr(rp, "retrain", fake_retrain)
    return log


def test_runs_etl_then_retrains_on_that_batch(calls, caplog):
    with caplog.at_level(logging.INFO, logger="pipeline"):
        rp.logger.propagate = True
        try:
            assert rp.main(["--source", "https://x.com/d", "--batch", "B7"]) == 0
        finally:
            rp.logger.propagate = False
    assert calls["steps"] == ["settings", "mlflow", "etl", "read", "train"]
    assert calls["etl_args"][0]["orders"] == "https://x.com/d/orders_B7.csv" and calls["etl_args"][1] == "B7"
    assert calls["read_args"] == ("B7", True) and calls["train_args"][0] == "B7"
    text = caplog.text
    assert "ETL batch 'B7': success" in text and "within the 180s limit" in text and "total" in text


def test_failed_etl_means_no_retraining(calls, monkeypatch):
    def broken(*a, **k):
        calls["steps"].append("etl")
        raise RuntimeError("HTTP 404")

    monkeypatch.setattr(rp, "run_etl", broken)
    assert rp.main(["--source", "s", "--batch", "B"]) == 1
    assert calls["steps"] == ["settings", "mlflow", "etl"]


def test_missing_settings_stop_before_downloading(calls, monkeypatch):
    def no_settings():
        raise RuntimeError("Missing database settings: DB_HOST")

    monkeypatch.setattr(rp, "get_settings", no_settings)
    assert rp.main(["--source", "s", "--batch", "B"]) == 1
    assert "etl" not in calls["steps"]


def test_batch_already_loaded_still_retrains(calls):
    calls["etl_result"] = ETLResult(batch_id="B", status="skipped")
    assert rp.main(["--source", "s", "--batch", "B"]) == 0
    assert calls["steps"][-1] == "train"


def test_a_failed_member_fails_the_pipeline(calls):
    calls["train_result"] = RetrainResult(batch="B", runs={"A/m": {}}, failed={"Ben": "ValueError: x"})
    assert rp.main(["--source", "s", "--batch", "B"]) == 1


def test_options_reach_the_steps(calls):
    rp.main(["--customers", "c.csv", "--orders", "o.csv", "--labels", "l.csv", "--batch-id", "Z", "--force",
             "--members", "A", "C", "--no-plots", "--sequential", "--no-cache"])
    assert calls["etl_args"] == ({"customers": "c.csv", "orders": "o.csv", "labels": "l.csv"}, "Z", True)
    assert calls["read_args"] == ("Z", False)
    assert calls["train_args"] == ("Z", ["A", "C"], False, False)


def test_argument_rules():
    with pytest.raises(SystemExit):
        rp.parse_args(["--source", "s"])
    with pytest.raises(SystemExit):
        rp.parse_args(["--customers", "a", "--orders", "b"])
