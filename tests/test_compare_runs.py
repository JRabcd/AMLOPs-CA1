"""Unit tests for the model comparison table (PBI-08.1). Uses a local MLflow store, no server."""
import time

import mlflow
import pandas as pd
import pytest

from src import compare_runs as cr
from src import train_common as tc


def log_run(member, model, batch, pr_auc, status="FINISHED"):
    with mlflow.start_run(run_name=f"etl_{batch}_{member}_{model}", tags={"member": member, "model": model, "batch": batch}):
        mlflow.log_metrics({"test_pr_auc": pr_auc, "test_precision": 0.5, "test_recall": 0.6, "test_f1": 0.55,
                            "test_false_alarm_rate": 0.09, "threshold": 0.3, "val_pr_auc": pr_auc + 0.05})
        if status != "FINISHED":
            mlflow.end_run(status=status)
    time.sleep(0.01)  # distinct start times


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "load_dotenv", lambda: None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    monkeypatch.chdir(tmp_path)
    tc.connect_mlflow()
    log_run("QianYunyang", "lgbm_v1", "base", 0.538)
    log_run("JianRong", "xgb_v2", "base", 0.544)
    log_run("ZhengHao", "logreg_v1", "base", 0.465)
    log_run("QianYunyang", "lgbm_v2", "base", 0.520)
    log_run("QianYunyang", "lgbm_v2", "base", 0.575)       # re-run with the same name: newest wins
    log_run("ZhengHao", "histgb_v2", "base", 0.99, status="FAILED")
    log_run("QianYunyang", "lgbm_v2", "B2", 0.580)        # newer batch
    yield tmp_path
    mlflow.set_tracking_uri(None)


def test_default_is_the_most_recent_batch(store, capsys):
    assert cr.main([]) == 0
    out = capsys.readouterr().out
    assert "batch 'B2': 1 runs" in out and "etl_B2_QianYunyang_lgbm_v2" in out and "etl_base_" not in out


def test_one_batch_ranked_by_pr_auc_without_failed_or_duplicate_runs(store):
    runs, batch = cr.pick_batch(cr.fetch_runs(), "base")
    table = cr.comparison_table(runs)
    assert batch == "base"
    assert list(table["Model"]) == ["lgbm_v2", "xgb_v2", "lgbm_v1", "logreg_v1"]
    assert list(table["Rank"]) == [1, 2, 3, 4]
    assert table.loc[0, "Test PR-AUC"] == pytest.approx(0.575)   # newest of the two lgbm_v2 runs
    assert "histgb_v2" not in set(table["Model"])                 # failed run left out


def test_markdown_and_csv(store, capsys):
    csv = store / "runs.csv"
    assert cr.main(["--batch", "base", "--csv", str(csv)]) == 0
    out = capsys.readouterr().out
    assert "| Rank | Run | Member |" in out and "| 1 | etl_base_QianYunyang_lgbm_v2 |" in out
    assert "| 9.0% |" in out                                       # false-alarm rate as a percentage
    assert len(pd.read_csv(csv)) == 4


def test_all_batches(store, capsys):
    assert cr.main(["--all"]) == 0
    assert "all batches: 5 runs" in capsys.readouterr().out


def test_unknown_batch_lists_the_real_ones(store, capsys):
    assert cr.main(["--batch", "nope"]) == 1
    assert "Batches in MLflow: B2, base" in capsys.readouterr().err


def test_empty_experiment(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tc, "load_dotenv", lambda: None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'e.db').as_posix()}")
    monkeypatch.chdir(tmp_path)
    try:
        assert cr.main([]) == 1
        assert "No finished runs" in capsys.readouterr().err
    finally:
        mlflow.set_tracking_uri(None)
