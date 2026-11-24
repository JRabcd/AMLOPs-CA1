"""Manual real-data integration check; run as ``python -m tests.smoke_transform``.

Set DATA_DIR to the CSV directory (for example, DATA_DIR=temp); default: data.
Reads the full local CSVs before selecting up to 1,000 labels, so this is not
a lightweight unit test. Run explicitly from the repository root with the
datasets available. This module performs file I/O at import time.
Sampling joins use raw keys before cleaning, so inconsistent key formatting
can omit matches. Very small or single-class samples may fail stratification
or the requirement that class weights include both classes.

The row-based stratified split checks pipeline compatibility only. For model
evaluation, use customer groups (see DI-02) and verify that customer snapshots
contain only information available at each order's prediction time.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.cleaning import clean_batch
from src.transform import (
    build_model_features,
    calculate_class_weights,
    prepare_features,
)

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))

# Load actual CSVs
customers = pd.read_csv(DATA_DIR / "customers_base.csv")
orders = pd.read_csv(DATA_DIR / "orders_base.csv")
labels = pd.read_csv(DATA_DIR / "labels_base.csv")

print("Raw shapes:")
print("Customers:", customers.shape)
print("Orders:", orders.shape)
print("Labels:", labels.shape)

# Sample labels first, then select matching records.
# This preserves the relationships between tables.
labels_sample = labels.sample(n=min(1000, len(labels)), random_state=42)

order_keys = labels_sample[["country_code", "order_id"]].drop_duplicates()

orders_sample = orders.merge(
    order_keys,
    on=["country_code", "order_id"],
    how="inner",
)

customer_keys = labels_sample[["country_code", "customer_id"]].drop_duplicates()

customers_sample = customers.merge(
    customer_keys,
    on=["country_code", "customer_id"],
    how="inner",
)

# Clean the sample
cleaned = clean_batch(
    customers_sample,
    orders_sample,
    labels_sample,
)

# Join and engineer features
features = prepare_features(
    cleaned.customers,
    cleaned.orders,
    cleaned.labels,
)

print("\nPrepared features:", features.shape)

# prepare_features excludes orphans by key; retain valid rows with missing dates.

# Split before fitting preprocessing; this is not a customer-isolated split.
train_df, test_df = train_test_split(
    features,
    test_size=0.2,
    random_state=42,
    stratify=features["is_fraud"],
)

X_train, X_test, artifacts = build_model_features(train_df, test_df)

print("\nMODEL FEATURE COLUMNS:")
print(X_train.columns.tolist())

weights = calculate_class_weights(train_df["is_fraud"])

print("\nX_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Class weights:", weights)

assert X_train.columns.tolist() == X_test.columns.tolist()
assert np.isfinite(X_train.to_numpy()).all()
assert np.isfinite(X_test.to_numpy()).all()
excluded = {
    "country_code",
    "order_id",
    "customer_id",
    "is_fraud",
    "order_date",
    "first_order_datetime",
    "refund_value",
    "refund_value_log1p",
    "refund_ratio",
}
assert excluded.isdisjoint(X_train.columns)
assert excluded.isdisjoint(X_test.columns)

print("\nSMOKE TEST PASSED")
