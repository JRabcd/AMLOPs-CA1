# PBI-07C: Zheng Hao

Zheng Hao's two models are a class-weighted logistic regression baseline
(`logreg_v1`) and HistGradientBoosting (`histgb_v2`). Both use the team's
`src/train_common.py` and `src/transform.py` unchanged.

## Setup and training

From the repository root, install the project in your virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you do not already have a local configuration.
Set `MLFLOW_TRACKING_URI` to the team's MLflow server URL and `DATA_DIR` to the
folder containing `customers_base.csv`, `orders_base.csv`, and `labels_base.csv`.
Use an absolute data path if your IDE launches from a different working folder.
Both entry points read `.env`; existing shell variables take precedence.

```powershell
$env:PYTHONIOENCODING = "utf-8"  # MLflow prints Unicode run links on Windows
.\.venv\Scripts\python.exe -m src.train_memberC --batch base
```

To verify uploads through the deployment entry point, run:

```powershell
.\.venv\Scripts\python.exe -m deploy_mlflow_aws.modelC_test --batch base
```

Both commands train the same two models. The shared loader reads `*_base.csv`;
`--batch` currently labels the runs and does not select different CSV files.
The experiment is `spbuy-fraud`; runs are named
`etl_base_ZhengHao_logreg_v1` and `etl_base_ZhengHao_histgb_v2`.

For local experiments, override the server setting in the same shell:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
.\.venv\Scripts\python.exe -m src.train_memberC --batch base
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Evaluation

The newest 20% of orders form the test set. Validation uses 15% of the remaining
customers, with no customer overlap between training and validation. Preprocessing
is fitted on training rows only. The F1 threshold is selected on validation;
reported test precision, recall and F1 use that threshold. Seed: 42.

The merged transformation excludes orphan labels from training and drops
current-order refund features (`refund_value`, its log, and `refund_ratio`) from
model inputs. Customer history snapshots still require confirmation that their
values were available at prediction time.

Each MLflow run stores parameters, validation/test metrics, threshold, fit time,
a confusion matrix, a PR curve, preprocessing artifacts, and the model.

## Verified AWS results (25 November 2026)

Both runs below were read back from the Elastic Beanstalk server as `FINISHED`,
with model status `READY` and all three supporting artifacts present. Each model
trained on 645,942 rows with 42 features.

| Metric | Logistic regression v1 | HistGradientBoosting v2 |
|---|---:|---:|
| Validation PR-AUC | 0.4366 | 0.6138 |
| Test PR-AUC | 0.4652 | 0.5419 |
| Test precision | 0.4161 | 0.5078 |
| Test recall | 0.6232 | 0.6235 |
| Test F1 | 0.4990 | 0.5597 |
| Selected threshold | 0.7081 | 0.3048 |
| Test false positives | 20,816 | 14,387 |
| Fit time (seconds) | 2.38 | 10.84 |

- Logistic regression run: `2c602bad190e4fdcb4d774ffe24ff627`;
  model: `m-a8de25ebac2d4a6cbdd2823736d8200f`.
- HistGradientBoosting run: `58daa7cb49a942f58b5434ae92a4a1fe`;
  model: `m-0a18acc7704e48ff87ae01b5c6ed7141`.

### Diagnosis and change from v1 to v2

V1 provides an interpretable linear baseline, but cannot directly represent
nonlinear feature interactions. Its class weighting also makes the default 0.5
threshold flag many genuine orders: test precision was 0.3114 and recall 0.8050
at that threshold. Validation selected 0.7081 instead.

V2 switches to HistGradientBoosting to capture nonlinear relationships. It uses
200 iterations, 63 leaf nodes, a minimum of 200 samples per leaf, L2
regularization of 5, and disabled early stopping so validation is reserved for
threshold selection. The selected threshold was 0.3048.

Compared with v1 at their selected thresholds, v2 improved test PR-AUC by 0.0767
and F1 by 0.0607. Recall stayed near 0.623 while precision rose from 0.4161 to
0.5078, producing 6,429 fewer false positives. Training took longer. These are
base-data results; performance on a new batch remains to be measured. Registry
and champion selection belong to the later team comparison.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pip check
```

PBI-07C tests verify both model types can log and reload through a temporary local
MLflow store. They use synthetic data, require no CSVs, and do not contact AWS.
The team's shared training and transformation tests cover splitting, thresholds,
and preprocessing.
