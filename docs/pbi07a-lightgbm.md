# PBI-07a: LightGBM models (Qian Yunyang)

Code: `src/train_memberA.py` (two versions) and `src/train_common.py` (data, split, metrics and MLflow logging shared by the team).

```bash
python -m src.train_memberA            # needs MLFLOW_TRACKING_URI; reads the base CSVs from data/ or DATA_DIR
```

## Results on the base data

Test set = newest 20% of orders (4 Apr to 11 May 2023, 12.5% fraud). Threshold chosen on validation (15% of the remaining customers).

| | v1 | v2 |
|---|---|---|
| Change | class-weighted baseline | no class weights, regularised, `order_month` feature |
| Validation PR-AUC | 0.610 | 0.647 |
| **Test PR-AUC** | **0.538** | **0.575** |
| Threshold | 0.769 | 0.299 |
| Precision / recall / F1 | 0.503 / 0.626 / 0.558 | 0.503 / 0.687 / 0.580 |
| Genuine orders wrongly flagged | 14,711 (8.9%) | 16,168 (9.7%) |

Random guessing scores a PR-AUC of 0.125 here. Accuracy is not used: always guessing "not fraud" scores about 89%.

## Diagnosing v1

1. **Scores too high.** Class weights push the average predicted risk to 0.295 when only 12.5% of orders are fraud. At the usual 0.5 cut-off that flags 16.6% of genuine orders, with a precision of 0.41.
2. **Errors depend on country.** Recall is 0.76 in PK but 0.42 in PH and TH. PH precision is only 0.23. MY is accurate (1.2% false alarms) because fraud there is rare.
3. **Fraud changes over time, differently by country** (fraud rate by month):

| | Nov | Dec | Jan | Feb | Mar | Apr | May |
|---|---|---|---|---|---|---|---|
| PK | 0.200 | 0.183 | 0.203 | 0.225 | **0.337** | 0.298 | 0.249 |
| PH | 0.105 | 0.101 | 0.091 | 0.086 | 0.072 | 0.063 | 0.078 |
| BD | 0.147 | 0.270 | 0.294 | 0.287 | 0.314 | 0.352 | 0.441 |

## From v1 to v2

| Step (each row trained separately) | Validation | Test |
|---|---|---|
| v1: class-weighted | 0.610 | 0.538 |
| drop class weights | 0.614 | 0.540 |
| + regularisation | 0.615 | 0.543 |
| v1 + `order_month` only | 0.640 | 0.569 |
| **v2: all three** | **0.647** | **0.575** |

Dropping class weights barely changes ranking, but it makes the scores sensible (average 0.122 against 0.125 actual before the month feature). Nearly all of the gain in PR-AUC comes from `order_month`.

## Limits

- **`order_month` is drift, not a fraud cause.** It lets the model learn the shifts above. May was never in training, so later months are unproven. It helped at three different cut-off dates (+0.007, +0.017, +0.028), so it is not one lucky split. With one year of data, month and calendar time can't be told apart.
- **v2 is not better everywhere.** PK recall rises from 0.76 to 0.88, but PK false alarms rise from 20% to 34%. PH false alarms fall from 10% to 5%, with recall falling from 0.42 to 0.36. Overall v2 flags slightly more genuine orders than v1.
- **Test results were looked at several times while choosing settings,** so treat them as slightly optimistic. The next unseen batch is the real test.
- About 46,000 customers appear in both train and test, as they would in real use.
- The customer `last_50days` columns hold one value per customer. If that snapshot was taken after some orders, all scores here are too optimistic. The data can't settle this.
- Not tried yet: training on recent months only, and a separate threshold per country.
