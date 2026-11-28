"""PBI-07a, Qian Yunyang: LightGBM, two versions.

    python -m src.train_memberA [--batch base]

v1 is the class-weighted baseline. v2 follows from diagnosing v1 (docs/pbi07a-lightgbm.md): no class
weights, more regularisation, and an order_month feature.
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import pandas as pd

from src.train_common import SEED, connect_mlflow, load_frame, make_matrices, split, train_and_log

MEMBER = "QianYunyang"
COMMON = dict(n_estimators=300, learning_rate=0.1, num_leaves=63)
V1 = dict(**COMMON, class_weight="balanced")
V2 = dict(**COMMON, min_child_samples=200, reg_lambda=5)


def add_order_month(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(order_month=frame["order_date"].dt.month.astype(float))


def build(params: dict) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**params, random_state=SEED, n_jobs=-1, verbose=-1)


def train(frame: pd.DataFrame, batch: str = "base", plots: bool = True) -> dict[str, dict]:
    """Train and log both versions on an already loaded frame (reused by retraining, PBI-10)."""
    parts = split(frame)
    return {
        "lgbm_v1": train_and_log("lgbm_v1", build(V1), V1, make_matrices(frame, *parts), MEMBER, batch, plots),
        "lgbm_v2": train_and_log("lgbm_v2", build(V2), V2, make_matrices(add_order_month(frame), *parts), MEMBER, batch, plots),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train Qian Yunyang's LightGBM models and log them to MLflow.")
    ap.add_argument("--batch", default="base", help="Batch name used in the MLflow run names")
    args = ap.parse_args(argv)
    connect_mlflow()  # fail before the slow part if MLflow isn't configured
    train(load_frame(), batch=args.batch)


if __name__ == "__main__":
    main()
