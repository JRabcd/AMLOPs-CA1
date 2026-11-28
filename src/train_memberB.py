
"""PBI-07b, Jian Rong: XGBoost. Run: python -m src.train_memberB [--batch base]"""

import argparse

import xgboost as xgb

from src.train_common import (
    SEED,
    load_frame,
    make_matrices,
    split,
    train_and_log,
    connect_mlflow
)


# Model configurations

V1_BASE = dict(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1
)

V2 = dict(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.1,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5
)


# Function called by the automated retraining pipeline

def train(frame, batch="base", plots=True):

    # Split data and prepare feature matrices
    data = make_matrices(frame, *split(frame))

    # Get training labels only
    y = data[0][1]

    # Calculate class weight for v1
    v1 = {
        **V1_BASE,
        "scale_pos_weight": float(
            (y == 0).sum() / (y == 1).sum()
        )
    }

    # Train and log both models
    return {
        name: train_and_log(
            name,
            xgb.XGBClassifier(
                **p,
                tree_method="hist",
                n_jobs=-1,
                random_state=SEED
            ),
            p,
            data,
            "JianRong",
            batch,
            plots
        )
        for name, p in (
            ("xgb_v1", v1),
            ("xgb_v2", V2)
        )
    }


# Function for running this file manually

def main(batch="base"):

    # Connect to the team's MLflow server
    connect_mlflow()

    # Load local CSV data and train models
    train(load_frame(), batch)


# Execute main only when this file is run directly

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="base")

    main(ap.parse_args().batch)
