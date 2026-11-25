"""Shared training code for PBI-07, so every member's models use the same data, split and metrics.

Rules built in here (reasons in docs/pbi07a-lightgbm.md):
  * test set = newest 20% of orders; validation = 15% of the remaining customers (no customer in
    both train and validation);
  * preprocessing is fitted on training rows only;
  * the decision threshold is chosen on validation, never on test;
  * orphan labels (no matching order or customer) are left out of training, as agreed in PR #111.

MLflow address: set MLFLOW_TRACKING_URI in the environment or .env. For a local try-out use
sqlite:///mlflow.db.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from src.cleaning import clean_batch
from src.db.config import load_dotenv
from src.transform import build_model_features, prepare_features

EXPERIMENT = "spbuy-fraud"
SEED = 42
TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.15


def load_frame() -> pd.DataFrame:
    """Cleaned and feature-engineered rows from the base CSVs in DATA_DIR (default data/).

    PBI-10 replaces this with a read from RDS; everything below only needs the returned frame.
    """
    folder = Path(os.environ.get("DATA_DIR", "data"))
    batch = clean_batch(*(pd.read_csv(folder / f"{t}_base.csv", dtype=str) for t in ("customers", "orders", "labels")))
    return drop_unusable(prepare_features(batch.customers, batch.orders, batch.labels))


def drop_unusable(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rows that have an order date (needed for the split) and customer history."""
    keep = frame["order_date"].notna() & frame["num_orders_last_50days"].notna()
    return frame[keep].reset_index(drop=True)


def split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row positions for train, validation and test."""
    order = np.argsort(frame["order_date"].to_numpy(), kind="stable")
    cut = int(len(order) * (1 - TEST_FRACTION))
    older, test = order[:cut], order[cut:]
    splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_FRACTION, random_state=SEED)
    tr, va = next(splitter.split(older, groups=frame["customer_id"].iloc[older]))
    return older[tr], older[va], test


def make_matrices(frame: pd.DataFrame, train, validation, test):
    """Feature matrices and targets. Validation and test are transformed together so columns match."""
    held = np.concatenate([validation, test])
    X_train, X_held, artifacts = build_model_features(frame.iloc[train], frame.iloc[held])
    n = len(validation)
    y = [frame["is_fraud"].iloc[i].astype(int).to_numpy() for i in (train, validation, test)]
    return (X_train, y[0]), (X_held.iloc[:n], y[1]), (X_held.iloc[n:], y[2]), artifacts


def best_threshold(y_true, proba) -> float:
    """Threshold with the highest F1. Call it on validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def score(y_true, proba, threshold: float) -> dict[str, float]:
    """Metrics at one threshold, including false alarms on genuine orders."""
    flagged = (proba >= threshold).astype(int)
    genuine = np.asarray(y_true) == 0
    false_alarms = int((flagged[genuine] == 1).sum())
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, flagged, zero_division=0),
        "recall": recall_score(y_true, flagged),
        "f1": f1_score(y_true, flagged),
        "false_alarms": false_alarms,
        "false_alarm_rate": false_alarms / max(int(genuine.sum()), 1),
    }


def log_model(model) -> None:
    """Each library needs its own MLflow flavour: mlflow.sklearn rejects LightGBM and XGBoost objects."""
    kind = type(model).__module__.split(".")[0]
    if kind == "lightgbm":
        import mlflow.lightgbm

        mlflow.lightgbm.log_model(model, name="model")
    elif kind == "xgboost":
        import mlflow.xgboost

        mlflow.xgboost.log_model(model, name="model")
    else:
        import mlflow.sklearn

        # MLflow 3 blocks HistGradientBoosting's tree type unless it is trusted (it is our own model).
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=[
            "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"])


def connect_mlflow() -> None:
    """Point MLflow at the team server, refusing to fall back silently to a local folder."""
    load_dotenv()
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("Set MLFLOW_TRACKING_URI (environment or .env), or use sqlite:///mlflow.db to try it locally.")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)


def train_and_log(name: str, model, params: dict, data, member: str, batch: str = "base", plots: bool = True) -> dict:
    """Fit, choose the threshold on validation, score on test, log everything to MLflow."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test), artifacts = data
    run_name = f"etl_{batch}_{member}_{name}"
    with mlflow.start_run(run_name=run_name, tags={"member": member, "batch": batch, "model": name}):
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        p_val, p_test = model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1]
        threshold = best_threshold(y_val, p_val)
        metrics = {f"val_{k}": v for k, v in score(y_val, p_val, threshold).items()}
        metrics |= {f"test_{k}": v for k, v in score(y_test, p_test, threshold).items()}
        metrics |= {f"test_at_0.5_{k}": v for k, v in score(y_test, p_test, 0.5).items() if k in ("precision", "recall", "f1", "false_alarm_rate")}
        metrics |= {"threshold": threshold, "fit_seconds": fit_seconds}

        mlflow.log_params({**params, "n_features": X_train.shape[1], "train_rows": len(X_train), "seed": SEED,
                           "split": f"time-test{TEST_FRACTION}/customer-val{VALIDATION_FRACTION}"})
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            joblib.dump(artifacts, Path(tmp) / "preprocessing.joblib")
            if plots:
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_predictions(y_test, (p_test >= threshold).astype(int), ax=ax)
                fig.savefig(Path(tmp) / "confusion_matrix.png")
                plt.close(fig)
                fig, ax = plt.subplots()
                PrecisionRecallDisplay.from_predictions(y_test, p_test, ax=ax)
                fig.savefig(Path(tmp) / "pr_curve.png")
                plt.close(fig)
            mlflow.log_artifacts(tmp)
        log_model(model)
    print(f"{run_name}: test PR-AUC {metrics['test_pr_auc']:.4f}, precision {metrics['test_precision']:.3f}, "
          f"recall {metrics['test_recall']:.3f}, F1 {metrics['test_f1']:.3f} at threshold {threshold:.3f}")
    return metrics
