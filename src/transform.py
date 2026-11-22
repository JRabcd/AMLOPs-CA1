"""
ST1519 Agile MLOps CA1 - PBI-04: Feature engineering & transformation
Owner: Zheng Hao (Group 3)

Provides feature engineering and transformation functions for customers,
orders, and labels datasets cleaned by the PBI-03 module. Applies PBI-02 EDA
findings to create derived features and prepare numerical model inputs.

Create deterministic features before splitting, then fit encoding, imputation,
and class weights using training data only. Customer activity must describe
history available at each order's prediction time to avoid temporal leakage.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight


def join_datasets(
    customers: pd.DataFrame, orders: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Left-join cleaned tables into one row per labelled order.

    Composite keys must be unique within each input table. Labels supply
    customer IDs; unmatched labels are retained with missing joined values.
    Print unmatched counts for diagnostics without modifying the inputs.
    """

    # Ensure join keys are unique in their respective tables
    if orders.duplicated(["country_code", "order_id"]).any():
        raise ValueError("Duplicate order keys detected.")

    if customers.duplicated(["country_code", "customer_id"]).any():
        raise ValueError("Duplicate customer keys detected.")

    if labels.duplicated(["country_code", "order_id"]).any():
        raise ValueError("Duplicate label keys detected.")

    # Join labels with orders
    merged = labels.merge(
        orders,
        on=["country_code", "order_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    # Keep track of labels without matching orders
    missing_orders = (merged["_merge"] == "left_only").sum()
    merged = merged.drop(columns=["_merge"])

    # Join the result with customers
    merged = merged.merge(
        customers,
        on=["country_code", "customer_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    # Keep track of labels without matching customers
    missing_customers = (merged["_merge"] == "left_only").sum()
    merged = merged.drop(columns=["_merge"])

    print(f"Missing matching orders: {missing_orders}")
    print(f"Missing matching customers: {missing_customers}")

    return merged


CATEGORICAL_COLUMNS = [
    "payment_method",
    "collect_type",
    "country_code",
]


def encode_categories(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    Fit one-hot encoding on training data and apply it to both copies.

    Missing categories become ``"unknown"``; unseen categories produce zeros
    for that field. Return both frames and the fitted encoder, preserving
    input indices and all non-categorical columns.
    """

    # Ensure required columns exist
    missing = [
        col
        for col in CATEGORICAL_COLUMNS
        if col not in train_df.columns or col not in test_df.columns
    ]

    if missing:
        raise ValueError(f"Missing categorical columns: {missing}")

    # Avoid modifying original DataFrames
    train = train_df.copy()
    test = test_df.copy()

    # Replace missing categories with a consistent token
    train_cat = train[CATEGORICAL_COLUMNS].fillna("unknown").astype(str)
    test_cat = test[CATEGORICAL_COLUMNS].fillna("unknown").astype(str)

    # Fit encoder using training data only
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int)

    train_encoded = encoder.fit_transform(train_cat)
    test_encoded = encoder.transform(test_cat)

    feature_names = encoder.get_feature_names_out(CATEGORICAL_COLUMNS)

    # Convert encoded arrays into DataFrames
    train_features = pd.DataFrame(
        train_encoded, columns=feature_names, index=train.index
    )

    test_features = pd.DataFrame(test_encoded, columns=feature_names, index=test.index)

    # Remove original text columns and append numerical features
    train = pd.concat([train.drop(columns=CATEGORICAL_COLUMNS), train_features], axis=1)

    test = pd.concat([test.drop(columns=CATEGORICAL_COLUMNS), test_features], axis=1)

    return train, test, encoder


SKEWED_NUMERIC_COLUMNS = [
    "num_orders_last_50days",
    "num_cancelled_orders_last_50days",
    "num_refund_orders_last_50days",
    "total_payment_last_50days",
    "num_associated_customers",
    "order_value",
    "num_items_ordered",
    "refund_value",
]


def transform_skewed_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append log1p features for the eight skewed columns identified in PBI-02.

    Preserve original columns and missing values in a copy. Coerce invalid
    numeric text to missing values and reject any negative numeric values.
    Refund features are retained here for analysis but excluded by
    ``build_model_features``.
    """

    transformed = df.copy()

    missing = [col for col in SKEWED_NUMERIC_COLUMNS if col not in transformed.columns]

    if missing:
        raise ValueError(f"Missing numerical columns: {missing}")

    for col in SKEWED_NUMERIC_COLUMNS:
        values = pd.to_numeric(transformed[col], errors="coerce")

        # Reject negatives because these features are non-negative by domain.
        if (values.dropna() < 0).any():
            raise ValueError(f"Negative values found in {col}")

        transformed[f"{col}_log1p"] = np.log1p(values.astype(float))

    return transformed


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append refund ratio, age since first order, and 50-day activity rates.

    Return a copy with zero denominators and invalid inputs mapped to missing
    values. Age is measured in fractional days since the first order, rather
    than account registration. Negative ages and rates above one are retained
    for upstream review. Current-order refund ratio is for analysis only.
    """

    result = df.copy()

    required_columns = [
        "refund_value",
        "order_value",
        "order_date",
        "first_order_datetime",
        "num_orders_last_50days",
        "num_cancelled_orders_last_50days",
        "num_refund_orders_last_50days",
    ]

    missing = [col for col in required_columns if col not in result.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 1. Refund ratio
    order_value = pd.to_numeric(result["order_value"], errors="coerce")
    refund_value = pd.to_numeric(result["refund_value"], errors="coerce")

    result["refund_ratio"] = refund_value / order_value.replace(0, np.nan)

    # 2. Account age in days
    order_dates = pd.to_datetime(result["order_date"], errors="coerce")
    first_order_dates = pd.to_datetime(result["first_order_datetime"], errors="coerce")

    result["account_age_days"] = (
        order_dates - first_order_dates
    ).dt.total_seconds() / 86400

    # 3. Recent cancellation and refund rates
    total_orders = pd.to_numeric(
        result["num_orders_last_50days"], errors="coerce"
    ).replace(0, np.nan)

    cancelled_orders = pd.to_numeric(
        result["num_cancelled_orders_last_50days"], errors="coerce"
    )

    refunded_orders = pd.to_numeric(
        result["num_refund_orders_last_50days"], errors="coerce"
    )

    result["cancellation_rate_50days"] = cancelled_orders / total_orders

    result["refund_rate_50days"] = refunded_orders / total_orders

    return result


def calculate_class_weights(y: pd.Series) -> dict:
    """
    Return balanced weights for training labels containing both 0 and 1.

    Reject missing labels and single-class inputs. Each weight equals the
    number of training rows divided by twice that class's row count.
    """

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    classes = np.unique(y)

    if len(classes) != 2 or set(classes) != {0, 1}:
        raise ValueError("Target must contain both classes: 0 and 1.")

    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y.to_numpy()
    )

    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def prepare_features(
    customers: pd.DataFrame, orders: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Join cleaned datasets and create numerical features.

    Expects inputs that have already passed through clean_batch().
    Does not fit encoders or perform a train/test split.

    The returned frame retains identifiers, labels, dates, and refund fields
    for analysis; use ``build_model_features`` before passing it to a model.
    This join requires a label-shaped order/customer key table, so unlabelled
    inference batches need caller-side adaptation; do not fabricate targets.
    """

    # Combine the three cleaned tables
    combined = join_datasets(customers, orders, labels)

    # Create derived features while raw columns are available
    combined = create_derived_features(combined)

    # Transform skewed numerical features
    combined = transform_skewed_numerics(combined)

    return combined


def build_model_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Prepare train/test features without fitting on test data.

    Return numerical matrices and fitted preprocessing artifacts.

    Exclude identifiers, target, dates, and current-order refund features.
    Keep numeric/boolean training columns, align test columns to that schema,
    and replace infinities with missing values before median imputation.
    All-missing training columns are retained and imputed with zero.

    Artifacts contain ``encoder``, ``imputer``, and ordered ``feature_columns``.
    For future batches, repeat the same cleaning and feature preparation,
    fill categorical nulls with ``"unknown"`` and cast to string, then call
    the saved encoder's ``transform``. Append its named columns, drop the
    original categorical fields, reindex to ``feature_columns``, replace
    infinities with missing values, and call the saved imputer's ``transform``.
    Rebuild the output frame using the batch index and saved column order.
    Calling this function again fits new artifacts; it is not transform-only.

    Customer aggregates and their derived features are safe only if their
    values were available at prediction time. Extra numeric input columns
    are included by default and require a separate leakage review.
    """

    # Exclude identifiers, target, raw dates and
    # potentially post-order refund information.
    excluded = [
        "country_code",  # Encoded separately
        "order_id",
        "customer_id",
        "is_fraud",
        "order_date",
        "first_order_datetime",
        "refund_value",
        "refund_value_log1p",
        "refund_ratio",
    ]

    # Encode categories using training data only
    train_encoded, test_encoded, encoder = encode_categories(train_df, test_df)

    # Remove unwanted columns
    train_features = train_encoded.drop(columns=excluded, errors="ignore")
    test_features = test_encoded.drop(columns=excluded, errors="ignore")

    # Keep numerical and boolean columns only
    train_features = train_features.select_dtypes(include=["number", "bool"])

    # Ensure identical columns and ordering
    test_features = test_features.reindex(columns=train_features.columns)

    # Replace infinity with missing values
    train_features = train_features.replace([np.inf, -np.inf], np.nan)
    test_features = test_features.replace([np.inf, -np.inf], np.nan)

    # Fit missing-value handling on training data only
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)

    train_array = imputer.fit_transform(train_features)
    test_array = imputer.transform(test_features)

    X_train = pd.DataFrame(
        train_array, columns=train_features.columns, index=train_features.index
    )

    X_test = pd.DataFrame(
        test_array, columns=train_features.columns, index=test_features.index
    )

    artifacts = {
        "encoder": encoder,
        "imputer": imputer,
        "feature_columns": train_features.columns.tolist(),
    }

    return X_train, X_test, artifacts
