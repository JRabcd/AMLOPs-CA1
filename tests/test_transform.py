"""
Unit tests for src/transform.py
ST1519 Agile MLOps CA1 - PBI-04 (Feature engineering & transformation)

Uses synthetic data to test dataset joins, feature calculations, categorical
encoding, class weights, and preprocessing fitted on training data only.
"""

import numpy as np
import pandas as pd
import pytest

from src.transform import (
    build_model_features,
    calculate_class_weights,
    create_derived_features,
    encode_categories,
    join_datasets,
    prepare_features,
    transform_skewed_numerics,
)


def test_join_datasets():
    # Two customers
    customers = pd.DataFrame(
        {
            "country_code": ["MY", "PH"],
            "customer_id": ["C001", "C002"],
            "mobile_verified": [True, False],
        }
    )

    # Two orders
    orders = pd.DataFrame(
        {
            "country_code": ["MY", "PH"],
            "order_id": ["O001", "O002"],
            "order_value": [100.0, 250.0],
        }
    )

    # Two fraud labels
    labels = pd.DataFrame(
        {
            "country_code": ["MY", "PH"],
            "order_id": ["O001", "O002"],
            "customer_id": ["C001", "C002"],
            "is_fraud": [0, 1],
        }
    )

    result = join_datasets(customers, orders, labels)

    assert len(result) == 2
    assert result["order_value"].tolist() == [100.0, 250.0]
    assert result["mobile_verified"].tolist() == [True, False]
    assert result["is_fraud"].tolist() == [0, 1]


def test_duplicate_order_keys_rejected():
    customers = pd.DataFrame(
        {
            "country_code": ["MY"],
            "customer_id": ["C001"],
        }
    )

    orders = pd.DataFrame(
        {
            "country_code": ["MY", "MY"],
            "order_id": ["O001", "O001"],
        }
    )

    labels = pd.DataFrame(
        {
            "country_code": ["MY"],
            "order_id": ["O001"],
            "customer_id": ["C001"],
        }
    )

    try:
        join_datasets(customers, orders, labels)
        assert False, "Expected ValueError for duplicate order keys"
    except ValueError as exc:
        assert "Duplicate order keys" in str(exc)


def test_unmatched_records_preserved(caplog):
    customers = pd.DataFrame(
        {
            "country_code": ["MY"],
            "customer_id": ["C001"],
            "mobile_verified": [True],
        }
    )

    orders = pd.DataFrame(
        {
            "country_code": ["MY"],
            "order_id": ["O001"],
            "order_value": [100.0],
        }
    )

    labels = pd.DataFrame(
        {
            "country_code": ["MY", "MY"],
            "order_id": ["O001", "O999"],
            "customer_id": ["C001", "C999"],
            "is_fraud": [0, 1],
        }
    )

    result = join_datasets(customers, orders, labels)

    assert len(result) == 2

    missing = result[result["order_id"] == "O999"].iloc[0]

    assert pd.isna(missing["order_value"])
    assert pd.isna(missing["mobile_verified"])
    assert missing["is_fraud"] == 1
    assert "Missing matching orders: 1" in caplog.text
    assert "Missing matching customers: 1" in caplog.text


def test_encode_categories():
    train = pd.DataFrame(
        {
            "payment_method": ["credit card", "paypal"],
            "collect_type": ["delivery", "pickup"],
            "country_code": ["MY", "PH"],
            "is_fraud": [0, 1],
        }
    )

    test = pd.DataFrame(
        {
            "payment_method": ["unknown new method"],
            "collect_type": ["delivery"],
            "country_code": ["MY"],
            "is_fraud": [0],
        }
    )

    train_encoded, test_encoded, encoder = encode_categories(train, test)

    # Both datasets must have identical columns
    assert train_encoded.columns.tolist() == test_encoded.columns.tolist()

    # Original categorical columns should be removed
    assert "payment_method" not in train_encoded.columns
    assert "collect_type" not in train_encoded.columns
    assert "country_code" not in train_encoded.columns

    # Unknown payment method should not crash
    payment_cols = [
        col for col in test_encoded.columns if col.startswith("payment_method_")
    ]
    assert test_encoded[payment_cols].iloc[0].sum() == 0

    # Target must remain unchanged
    assert train_encoded["is_fraud"].tolist() == [0, 1]


def test_transform_skewed_numerics():
    # Create all eight required numerical columns
    df = pd.DataFrame(
        {
            col: [0.0, 9.0, 99.0]
            for col in [
                "num_orders_last_50days",
                "num_cancelled_orders_last_50days",
                "num_refund_orders_last_50days",
                "total_payment_last_50days",
                "num_associated_customers",
                "order_value",
                "num_items_ordered",
                "refund_value",
            ]
        }
    )

    df["is_fraud"] = [0, 1, 0]

    result = transform_skewed_numerics(df)

    # All eight transformed columns must exist
    for col in df.columns:
        if col == "is_fraud":
            continue

        log_col = f"{col}_log1p"

        assert log_col in result.columns
        np.testing.assert_allclose(result[log_col], np.log1p(df[col]))

    # Original data and fraud labels remain unchanged
    pd.testing.assert_frame_equal(result[df.columns], df)


def test_transform_skewed_numerics_rejects_negative():
    df = pd.DataFrame(
        {
            col: [1.0, 2.0]
            for col in [
                "num_orders_last_50days",
                "num_cancelled_orders_last_50days",
                "num_refund_orders_last_50days",
                "total_payment_last_50days",
                "num_associated_customers",
                "order_value",
                "num_items_ordered",
                "refund_value",
            ]
        }
    )

    df.loc[0, "order_value"] = -10

    with pytest.raises(ValueError, match="Negative values"):
        transform_skewed_numerics(df)


def test_create_derived_features():
    df = pd.DataFrame(
        {
            "refund_value": [20.0, 0.0],
            "order_value": [100.0, 0.0],
            "order_date": ["2026-06-11", "2026-06-12"],
            "first_order_datetime": [
                "2026-06-01",
                "2026-06-10",
            ],
            "num_orders_last_50days": [10, 0],
            "num_cancelled_orders_last_50days": [2, 0],
            "num_refund_orders_last_50days": [3, 0],
            "is_fraud": [1, 0],
        }
    )

    result = create_derived_features(df)

    # Verify calculations for first record
    assert np.isclose(result.loc[0, "refund_ratio"], 0.2)
    assert np.isclose(result.loc[0, "account_age_days"], 10)
    assert np.isclose(result.loc[0, "cancellation_rate_50days"], 0.2)
    assert np.isclose(result.loc[0, "refund_rate_50days"], 0.3)

    # Zero denominators should produce missing values
    assert pd.isna(result.loc[1, "refund_ratio"])
    assert pd.isna(result.loc[1, "cancellation_rate_50days"])
    assert pd.isna(result.loc[1, "refund_rate_50days"])

    # Original target preserved
    assert result["is_fraud"].tolist() == [1, 0]


def test_calculate_class_weights():
    # 8 legitimate orders, 2 fraudulent orders
    y = pd.Series([0] * 8 + [1] * 2)

    weights = calculate_class_weights(y)

    assert np.isclose(weights[0], 0.625)
    assert np.isclose(weights[1], 2.5)

    # Fraud should receive a larger weight
    assert weights[1] > weights[0]


def test_negative_account_ages_retained_and_logged(caplog):
    df = pd.DataFrame(
        {
            "order_date": pd.to_datetime(["2026-06-01"] * 4),
            "first_order_datetime": pd.to_datetime(
                ["2026-06-03", "2026-06-01", "2026-05-30", None]
            ),
            "order_value": [100.0] * 4,
            "refund_value": [0.0] * 4,
            "num_orders_last_50days": [10] * 4,
            "num_cancelled_orders_last_50days": [0] * 4,
            "num_refund_orders_last_50days": [0] * 4,
        }
    )

    result = create_derived_features(df)

    np.testing.assert_allclose(result["account_age_days"], [-2, 0, 2, np.nan])
    assert "account_age_negative" not in result.columns
    assert "Negative account ages retained for upstream review: 1" in caplog.text


def test_calculate_class_weights_single_class():
    y = pd.Series([0, 0, 0, 0])

    with pytest.raises(ValueError, match="both classes"):
        calculate_class_weights(y)


def test_prepare_features_integration():
    customers = pd.DataFrame(
        {
            "country_code": ["MY"],
            "customer_id": ["C001"],
            "first_order_datetime": ["2026-01-01"],
            "num_orders_last_50days": [10],
            "num_cancelled_orders_last_50days": [2],
            "num_refund_orders_last_50days": [1],
            "total_payment_last_50days": [1000.0],
            "num_associated_customers": [3],
        }
    )

    orders = pd.DataFrame(
        {
            "country_code": ["MY"],
            "order_id": ["O001"],
            "order_date": ["2026-01-11"],
            "order_value": [100.0],
            "refund_value": [20.0],
            "num_items_ordered": [2],
            "payment_method": ["credit card"],
            "collect_type": ["delivery"],
        }
    )

    labels = pd.DataFrame(
        {
            "country_code": ["MY"],
            "order_id": ["O001"],
            "customer_id": ["C001"],
            "is_fraud": [1],
        }
    )

    result = prepare_features(customers, orders, labels)

    assert len(result) == 1
    assert result.loc[0, "is_fraud"] == 1

    assert np.isclose(result.loc[0, "refund_ratio"], 0.2)
    assert np.isclose(result.loc[0, "account_age_days"], 10)
    assert np.isclose(result.loc[0, "cancellation_rate_50days"], 0.2)

    assert np.isclose(result.loc[0, "order_value_log1p"], np.log1p(100))


def test_build_model_features():
    train = pd.DataFrame(
        {
            "country_code": ["MY", "PH"],
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C2"],
            "payment_method": ["credit card", "paypal"],
            "collect_type": ["delivery", "pickup"],
            "order_value": [100.0, np.nan],
            "refund_value": [20.0, 0.0],
            "refund_value_log1p": np.log1p([20.0, 0.0]),
            "refund_ratio": [0.2, np.nan],
            "order_date": pd.to_datetime(["2026-01-11", "2026-01-12"]),
            "first_order_datetime": pd.to_datetime(["2026-01-01", None]),
            "is_fraud": [0, 1],
        }
    )

    test = pd.DataFrame(
        {
            "country_code": ["MY"],
            "order_id": ["O3"],
            "customer_id": ["C3"],
            "payment_method": ["new method"],
            "collect_type": ["delivery"],
            "order_value": [np.nan],
            "refund_value": [10.0],
            "refund_value_log1p": np.log1p([10.0]),
            "refund_ratio": [0.5],
            "order_date": pd.to_datetime(["2026-01-13"]),
            "first_order_datetime": pd.to_datetime([None]),
            "is_fraud": [1],
        }
    )

    X_train, X_test, artifacts = build_model_features(train, test)

    # Identical feature structure
    assert X_train.columns.tolist() == X_test.columns.tolist()

    # No missing values
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()

    # Excluded fields cannot enter the model
    for col in [
        "country_code",
        "order_id",
        "customer_id",
        "is_fraud",
        "order_date",
        "first_order_datetime",
        "refund_value",
        "refund_value_log1p",
        "refund_ratio",
    ]:
        assert col not in X_train.columns
        assert col not in X_test.columns

    # Median imputation learned from training data
    assert X_test["order_value"].iloc[0] == 100.0

    # Fitted preprocessing objects returned
    assert "encoder" in artifacts
    assert "imputer" in artifacts


def test_prepare_features_excludes_orphans_but_retains_missing_features(caplog):
    customers = pd.DataFrame(
        {
            "country_code": ["MY"],
            "customer_id": ["C001"],
            "first_order_datetime": [pd.NaT],
            "num_orders_last_50days": [10],
            "num_cancelled_orders_last_50days": [2],
            "num_refund_orders_last_50days": [1],
            "total_payment_last_50days": [1000.0],
            "num_associated_customers": [3],
        }
    )
    orders = pd.DataFrame(
        {
            "country_code": ["MY", "MY"],
            "order_id": ["O001", "O003"],
            "order_date": pd.to_datetime(["2026-01-11", "2026-01-12"]),
            "order_value": [np.nan, 100.0],
            "refund_value": [0.0, 0.0],
            "num_items_ordered": [2, 1],
            "payment_method": ["credit card", "paypal"],
            "collect_type": ["delivery", "pickup"],
        }
    )
    # Missing order only, missing customer only, and missing both.
    labels = pd.DataFrame(
        {
            "country_code": ["MY"] * 4,
            "order_id": ["O001", "O002", "O003", "O004"],
            "customer_id": ["C001", "C001", "C999", "C999"],
            "is_fraud": [1, 0, 1, 0],
        }
    )

    result = prepare_features(customers, orders, labels)

    assert result["order_id"].tolist() == ["O001"]
    assert result["is_fraud"].tolist() == [1]
    assert result["first_order_datetime"].isna().all()
    assert result["account_age_days"].isna().all()
    assert result["order_value"].isna().all()
    assert "Missing matching orders: 2" in caplog.text
    assert "Missing matching customers: 2" in caplog.text
    # A record missing both matches must be counted only once.
    assert "Excluded orphan records from training extract: 3" in caplog.text

    X_train, X_test, _ = build_model_features(result, result.copy())
    assert len(X_train) == len(X_test) == 1
    assert np.isfinite(X_train.to_numpy()).all()
    assert np.isfinite(X_test.to_numpy()).all()
