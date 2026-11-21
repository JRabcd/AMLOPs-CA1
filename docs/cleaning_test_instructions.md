# Test Instructions for Data Cleaning Module (`src/cleaning.py`)

**Owner / Author:** Jian Rong (PBI-03)  
**Target Audience:** Zheng Hao (PBI-03.5 Verification & Downstream PBI-04 Integration)

---

## 1. Overview & Purpose

`src/cleaning.py` provides deterministic, idempotent, and vectorized data cleaning functions for `customers`, `orders`, and `labels` batches. It prepares raw batch CSVs for database ingestion (`docs/rds-schema.md`) and downstream feature transformation (`PBI-04`).

This guide explains how to test `src/cleaning.py` both programmatically (via Python) and via the Command Line Interface (CLI) on a simulated batch with injected data issues.

---

## 2. Test Files

The test files can be generated from the base files or any incoming `_XXX` batch. A test batch with intentionally injected anomalies includes:
- `temp/customers_TEST.csv`: ~20k rows with lowercase country codes (`" my "`), boolean tokens (`"yes"`, `"1"`), and exact duplicate rows.
- `temp/orders_TEST.csv`: ~20k rows with lowercase country codes (`"ph"`), category synonyms (`"on Delivery"`, `"Online Banking"`), mixed date formats (`"2023/Feb/15"`), composite key duplicates, and impossible values (`order_value = -25.50`, `num_items_ordered = 0`).
- `temp/labels_TEST.csv`: ~20k rows with lowercase country codes (`"pk"`).

*(Note: The `temp/` folder is git-ignored and will not pollute the repository).*

---

## 3. How to Run the Tests

### Option A: Via Python API (Import into ETL or script)

```python
import pandas as pd
from src.cleaning import clean_batch

# Read raw CSVs as strings (preserves leading zeros and raw representations)
customers_df = pd.read_csv("temp/customers_TEST.csv", dtype=str)
orders_df = pd.read_csv("temp/orders_TEST.csv", dtype=str)
labels_df = pd.read_csv("temp/labels_TEST.csv", dtype=str)

# Run cleaning pipeline
result = clean_batch(customers_df, orders_df, labels_df)

# Access cleaned DataFrames
clean_customers = result.customers
clean_orders = result.orders
clean_labels = result.labels
report = result.report

print(f"Cleaned {len(clean_customers):,} customers, {len(clean_orders):,} orders, {len(clean_labels):,} labels")
```

### Option B: Via Command Line Interface (CLI)

Run the standalone CLI module from the repository root:

```bash
python -m src.cleaning \
  --customers temp/customers_TEST.csv \
  --orders temp/orders_TEST.csv \
  --labels temp/labels_TEST.csv \
  --out temp/clean_out
```

---

## 4. Expected Output & Behavior

When executing Option B:
1. **Output Files Created**:
   - `temp/clean_out/customers_clean.csv`
   - `temp/clean_out/orders_clean.csv`
   - `temp/clean_out/labels_clean.csv`
   - `temp/clean_out/cleaning_report.json`
2. **Deduplication**:
   - Exact duplicate rows are detected and removed first.
   - Key duplicates on `(country_code, order_id)` or `(country_code, customer_id)` retain the last row (`keep='last'`).
3. **Canonical Normalization**:
   - `country_code`: 100% uppercased and stripped (`BD`, `MY`, `PH`, `PK`, `TH`).
   - `collect_type`: canonical lowercase (`delivery`, `pickup`).
   - `payment_method`: canonical lowercase with synonyms unified (`on delivery` -> `payment on delivery`).
   - `mobile_verified`: boolean values (`True`, `False`, `<NA>`).
   - `order_date`: 100% parsed into `datetime64[ns]` across formats (`%Y/%m/%d`, `%Y-%m-%d`, `%Y/%b/%d`).
4. **Data Type Compliance**:
   - Counts -> `Int64` (nullable integer).
   - Money -> `float64`.
   - Dates -> `datetime64[ns]`.
   - `is_fraud` -> `int64` (`0` or `1`).

---

## 5. Schema Failure Test (Expected Rejection)

To test the fail-fast schema safety check, drop any required column (e.g. `mobile_verified`) and run:

```python
import pandas as pd
from src.cleaning import clean_customers

bad_df = pd.read_csv("temp/customers_TEST.csv", dtype=str).drop(columns=["mobile_verified"])
clean_customers(bad_df)
```

**Expected Result:**  
Raises `ValueError: Schema validation failed for 'customers': missing column(s): ['mobile_verified']`.
