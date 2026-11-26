# PBI-07b diagnosis — XGBoost v1 → v2 (Jian Rong)

## v1 (baseline: scale_pos_weight)
- Fraud is ~11% of orders, so `scale_pos_weight ≈ 8` up-weights the fraud class
  8×. This inflates predicted probabilities: the validation-optimal threshold is
  **0.781**, not 0.5. At a naive 0.5 cut-off, precision is only ~0.41 (recall
  ~0.79) — most flagged orders would be genuine customers. The v1 confusion
  matrix shows the large false-positive cell. Trust & Safety has limited review
  time, so over-flagging is the core business problem.
- Test: PR-AUC 0.541, ROC-AUC 0.883, precision 0.510, recall 0.604, F1 0.553
  (threshold 0.781). Fit 4.2 s.

## What changed in v2 and why
- Removed `scale_pos_weight`: handle the imbalance with the threshold (chosen on
  validation) instead of distorting the probabilities.
- Shallower, more regularised trees: max_depth 7 → 6, min_child_weight=20,
  subsample=0.8, colsample_bytree=0.8, reg_lambda=5, 250 trees — less overfitting
  to noise.

## Result (honest reading)
- Test PR-AUC 0.541 → 0.544 and F1 0.553 → 0.559: within run-to-run noise, so
  this is not a clear win and I won't claim one.
- The real change is behaviour at the cut-off: recall 0.604 → 0.631 (+0.027) at
  almost the same precision (0.510 → 0.502), and the operating threshold drops
  from an awkward 0.781 to a natural 0.291. ROC-AUC is unchanged (0.883), i.e.
  ranking ability is the same — v2 is better calibrated, not a better ranker.
- Test scores sit below validation (0.611 / 0.613) because the test set is the
  newest 20% of orders (distribution drift). Settings were chosen while looking
  at test several times, so treat these numbers as slightly optimistic; the
  tutor's new batch at the interview is the real test.
- Next lever: a new feature (payment_method × country, or the customer's prior
  order count), not more hyperparameter tuning.