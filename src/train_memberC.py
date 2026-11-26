"""PBI-07C, Zheng Hao: logistic regression and HistGradientBoosting.

Run from the repository root: python -m src.train_memberC [--batch base].
"""

import argparse

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.train_common import (
    SEED,
    connect_mlflow,
    load_frame,
    make_matrices,
    split,
    train_and_log,
)

V1 = dict(class_weight="balanced", max_iter=500)
V2 = dict(
    max_iter=200,
    learning_rate=0.1,
    max_leaf_nodes=63,
    min_samples_leaf=200,
    l2_regularization=5,
    early_stopping=False,
)


def main(batch="base"):
    connect_mlflow()
    f = load_frame()
    data = make_matrices(f, *split(f))
    results = {}
    results["logreg_v1"] = train_and_log(
        "logreg_v1",
        make_pipeline(StandardScaler(), LogisticRegression(**V1, random_state=SEED)),
        V1,
        data,
        member="ZhengHao",
        batch=batch,
    )
    results["histgb_v2"] = train_and_log(
        "histgb_v2",
        HistGradientBoostingClassifier(**V2, random_state=SEED),
        V2,
        data,
        member="ZhengHao",
        batch=batch,
    )
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--batch",
        default="base",
        help="Run label; the shared loader currently reads *_base.csv",
    )
    main(ap.parse_args().batch)
