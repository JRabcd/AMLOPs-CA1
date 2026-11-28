"""Run Zheng Hao's real models against the configured MLflow server.

From the repository root, set MLFLOW_TRACKING_URI and run:
    python -m deploy_mlflow_aws.modelC_test --batch base
"""

import argparse

from src.train_memberC import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", default="base")
    args = ap.parse_args()
    main(batch=args.batch)
