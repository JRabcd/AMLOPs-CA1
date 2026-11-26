"""Training safeguards and local MLflow integration; no CSVs or AWS required."""

import numpy as np
import pandas as pd
import pytest
import mlflow
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import train_common as common
from src.train_memberC import SEED, V1, V2


@pytest.mark.parametrize("kind", ["logreg", "histgb"])
def test_models_log_and_reload_locally(tmp_path, monkeypatch, kind):
    monkeypatch.chdir(tmp_path)
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    )
    monkeypatch.setattr(common, "load_dotenv", lambda: None)
    common.connect_mlflow()
    X, y = make_classification(n_samples=160, n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=list("abcde"))
    data = (
        (X.iloc[:100], y[:100]),
        (X.iloc[100:130], y[100:130]),
        (X.iloc[130:], y[130:]),
        {"columns": list(X.columns)},
    )
    model = (
        make_pipeline(StandardScaler(), LogisticRegression(**V1, random_state=SEED))
        if kind == "logreg"
        else HistGradientBoostingClassifier(
            **{**V2, "max_iter": 3, "min_samples_leaf": 5}, random_state=SEED
        )
    )
    try:
        metrics = common.train_and_log(
            kind, model, {}, data, member="ZhengHao", batch="synthetic"
        )
        assert 0 <= metrics["test_pr_auc"] <= 1
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(common.EXPERIMENT)
        run = client.search_runs([experiment.experiment_id])[0]
        assert run.info.run_name == f"etl_synthetic_ZhengHao_{kind}"
        assert run.data.tags["batch"] == "synthetic"
        artifacts = {item.path for item in client.list_artifacts(run.info.run_id)}
        assert {
            "confusion_matrix.png",
            "pr_curve.png",
            "preprocessing.joblib",
        } <= artifacts
        models = client.search_logged_models(experiment_ids=[experiment.experiment_id])
        restored = mlflow.sklearn.load_model(f"models:/{models[0].model_id}")
        np.testing.assert_allclose(
            restored.predict_proba(X.iloc[130:]), model.predict_proba(X.iloc[130:])
        )
    finally:
        mlflow.set_tracking_uri(previous_uri)
