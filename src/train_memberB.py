"""PBI-07b, Jian Rong: XGBoost.  Run: python -m src.train_memberB [--batch base]"""
import argparse

import xgboost as xgb

from src.train_common import SEED, load_frame, make_matrices, split, train_and_log


def main(batch="base"):
    f = load_frame()
    data = make_matrices(f, *split(f))
    y = data[0][1]
    v1 = dict(n_estimators=300, max_depth=7, learning_rate=0.1,
              scale_pos_weight=float((y == 0).sum() / (y == 1).sum()))    # baseline, weighted
    v2 = dict(n_estimators=250, max_depth=6, learning_rate=0.1, min_child_weight=20,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=5)          # unweighted, shallower, regularised
    for name, p in (("xgb_v1", v1), ("xgb_v2", v2)):
        train_and_log(name, xgb.XGBClassifier(**p, tree_method="hist", n_jobs=-1, random_state=SEED), p,
                      data, member="JianRong", batch=batch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--batch", default="base")
    main(ap.parse_args().batch)