"""
Unit tests for src/cleaning.py
ST1519 Agile MLOps CA1 - PBI-03 (Reusable data-cleaning module)
"""

import numpy as np
import pandas as pd
import pytest

from src.cleaning import (
    CleanResult,
    DATE_FORMATS,
    FALSE_TOKENS,
    TRUE_TOKENS,
    VALID_COUNTRIES,
    clean_batch,
    clean_customers,
    clean_labels,
    clean_orders,
    coerce_numeric,
    drop_duplicates_by_key,
    normalize_category,
    normalize_country_code,
    parse_dates,
    to_boolean,
    validate_schema,
)


# ==============================================================================
# 1. SCHEMA VALIDATION TESTS
# ==============================================================================

def test_validate_schema_valid():
    df = pd.DataFrame(columns=[
        "country_code", "customer_id", "mobile_verified",
        "num_orders_last_50days", "num_cancelled_orders_last_50days",
        "num_refund_orders_last_50days", "total_payment_last_50days",
        "num_associated_customers", "first_order_datetime"
    ])
    validate_schema(df, "customers")  # Should not raise


def test_validate_schema_missing_column():
    df = pd.DataFrame(columns=["country_code", "customer_id"])
    with pytest.raises(ValueError, match="missing column"):
        validate_schema(df, "customers")


def test_validate_schema_extra_column():
    df = pd.DataFrame(columns=[
        "country_code", "order_id", "collect_type", "payment_method",
        "order_value", "num_items_ordered", "refund_value", "order_date",
        "unexpected_extra_col"
    ])
    with pytest.raises(ValueError, match="unexpected extra column"):
        validate_schema(df, "orders")


def test_validate_schema_unknown_table():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Unknown table"):
        validate_schema(df, "nonexistent_table")


# ==============================================================================
# 2. ATOMIC HELPER TESTS
# ==============================================================================

def test_normalize_country_code():
    s = pd.Series([" ph ", "my", "PK", "BD", "th", "invalid_cc", None, ""])
    cleaned, mod_count = normalize_country_code(s)
    
    assert cleaned.iloc[0] == "PH"
    assert cleaned.iloc[1] == "MY"
    assert cleaned.iloc[2] == "PK"
    assert cleaned.iloc[3] == "BD"
    assert cleaned.iloc[4] == "TH"
    assert pd.isna(cleaned.iloc[6])
    assert pd.isna(cleaned.iloc[7])
    assert mod_count == 4  # ph, my, th, and stripped ' ph '


def test_to_boolean_tokens():
    for tok in TRUE_TOKENS:
        s = pd.Series([tok])
        cleaned, fixed, unmappable = to_boolean(s)
        assert cleaned.iloc[0] is True or cleaned.iloc[0] == True
        assert unmappable == 0

    for tok in FALSE_TOKENS:
        s = pd.Series([tok])
        cleaned, fixed, unmappable = to_boolean(s)
        assert cleaned.iloc[0] is False or cleaned.iloc[0] == False
        assert unmappable == 0

    s_bad = pd.Series(["maybe", "unknown", None, "nan"])
    cleaned_bad, fixed, unmappable = to_boolean(s_bad)
    assert cleaned_bad.isna().all()
    assert unmappable == 2  # 'maybe' and 'unknown'


def test_normalize_category():
    s = pd.Series(["  on Delivery  ", "Online Banking", "CYBERSOURCE", "nan", None, ""])
    synonyms = {"on delivery": "payment on delivery"}
    cleaned, fixed = normalize_category(s, synonyms=synonyms, allow_null=True)
    
    assert cleaned.iloc[0] == "payment on delivery"
    assert cleaned.iloc[1] == "online banking"
    assert cleaned.iloc[2] == "cybersource"
    assert pd.isna(cleaned.iloc[3])
    assert pd.isna(cleaned.iloc[4])
    assert pd.isna(cleaned.iloc[5])


def test_parse_dates_mixed():
    s = pd.Series([
        "2023-04-09 14:30:00",
        "2023-04-09",
        "2023/04/09",
        "2023/Feb/09",
        "09/04/2023",
        "invalid_date",
        None
    ])
    dt, used, unparseable = parse_dates(s)
    
    assert pd.to_datetime("2023-04-09 14:30:00") == dt.iloc[0]
    assert pd.to_datetime("2023-04-09") == dt.iloc[1]
    assert pd.to_datetime("2023-04-09") == dt.iloc[2]
    assert pd.to_datetime("2023-02-09") == dt.iloc[3]
    assert pd.to_datetime("2023-04-09") == dt.iloc[4]
    assert pd.isna(dt.iloc[5])
    assert pd.isna(dt.iloc[6])
    assert unparseable == 1  # only 'invalid_date'


def test_coerce_numeric():
    # Float with positive constraint (e.g. order_value)
    s = pd.Series(["10.50", "0", "-5.0", "abc", None, ""])
    coerced, non_num, viol = coerce_numeric(s, is_integer=False, min_val=0, min_inclusive=False)
    assert coerced.iloc[0] == 10.50
    assert pd.isna(coerced.iloc[1])  # 0 is not allowed
    assert pd.isna(coerced.iloc[2])  # -5 is not allowed
    assert pd.isna(coerced.iloc[3])  # 'abc' is non-numeric
    assert non_num == 1
    assert viol == 2

    # Integer count with >= 0 constraint
    s_int = pd.Series(["5", "0", "1.5", "-2", "xyz"])
    coerced_int, non_num_int, viol_int = coerce_numeric(s_int, is_integer=True, min_val=0, min_inclusive=True)
    assert coerced_int.iloc[0] == 5
    assert coerced_int.iloc[0] == 5
    assert coerced_int.iloc[1] == 0
    assert pd.isna(coerced_int.iloc[2])  # non-integer
    assert pd.isna(coerced_int.iloc[3])  # negative
    assert pd.isna(coerced_int.iloc[4])  # non-numeric


def test_drop_duplicates_by_key():
    df = pd.DataFrame({
        "country_code": ["MY", "MY", "MY", "PH"],
        "order_id": ["o1", "o1", "o1", "o1"],
        "val": [10, 10, 20, 30]
    })
    # Row 0 and Row 1 are exact duplicates.
    # Row 1 and Row 2 have the same key ('MY', 'o1') with differing 'val' (20).
    deduped, exact, key = drop_duplicates_by_key(df, ["country_code", "order_id"])
    
    assert exact == 1
    assert key == 1
    assert len(deduped) == 2
    # Keep last retained row 2 (val=20) for ('MY', 'o1')
    assert deduped.loc[deduped["country_code"] == "MY", "val"].iloc[0] == 20
    assert deduped.loc[deduped["country_code"] == "PH", "val"].iloc[0] == 30


# ==============================================================================
# 3. END-TO-END TABLE CLEANING TESTS
# ==============================================================================

def test_clean_customers_end_to_end():
    raw_customers = pd.DataFrame({
        "country_code": [" my ", "PH", "invalid", "MY"],
        "customer_id": ["c1", "c2", "c3", "c1"],
        "mobile_verified": ["yes", "False", "1", "true"],
        "num_orders_last_50days": ["2", "0", "5", "3"],
        "num_cancelled_orders_last_50days": ["0", "0", "1", "0"],
        "num_refund_orders_last_50days": ["0", "0", "0", "0"],
        "total_payment_last_50days": ["45.50", "0.0", "-10.0", "60.0"],
        "num_associated_customers": ["1", "2", "1", "1"],
        "first_order_datetime": ["2022-01-01 10:00:00", "2023-05-10", "2021-12-12", "2022-01-01 10:00:00"]
    })

    cleaned, report = clean_customers(raw_customers)
    
    # 'invalid' country row is dropped
    # ('MY', 'c1') has key duplicate: row 0 vs row 3 -> keep last (row 3)
    assert len(cleaned) == 2
    assert set(cleaned["country_code"]) == {"MY", "PH"}
    assert cleaned.loc[cleaned["customer_id"] == "c1", "num_orders_last_50days"].iloc[0] == 3
    assert cleaned["mobile_verified"].dtype == "boolean"
    assert cleaned["num_orders_last_50days"].dtype == "Int64"
    assert cleaned["total_payment_last_50days"].dtype == "float64"
    assert pd.api.types.is_datetime64_any_dtype(cleaned["first_order_datetime"])


def test_clean_orders_end_to_end():
    raw_orders = pd.DataFrame({
        "country_code": ["ph", "MY", "PK", "TH"],
        "order_id": ["o1", "o2", "o3", "o4"],
        "collect_type": ["Delivery", "pickup", "delivery", "DELIVERY"],
        "payment_method": ["on Delivery", "Online Banking", "nan", "credit card"],
        "order_value": ["15.5", "0.0", "-5", "100.25"],  # 0.0 and -5 should become NaN
        "num_items_ordered": ["2", "1", "0", "1.5"],      # 0 and 1.5 should become pd.NA
        "refund_value": ["50.0", "0", "0", "0"],          # 50.0 > 15.5 kept raw (DI-38)
        "order_date": ["2023/04/09", "2023-04-09", "2023/Feb/09", "2023/12/31"]
    })

    cleaned, report = clean_orders(raw_orders)
    
    assert len(cleaned) == 4
    assert cleaned["country_code"].tolist() == ["PH", "MY", "PK", "TH"]
    assert cleaned["collect_type"].tolist() == ["delivery", "pickup", "delivery", "delivery"]
    assert cleaned["payment_method"].iloc[0] == "payment on delivery"
    assert cleaned["payment_method"].iloc[1] == "online banking"
    assert pd.isna(cleaned["payment_method"].iloc[2])  # 'nan' token -> pd.NA
    
    # Numeric violations set to NaN
    assert pd.isna(cleaned.loc[cleaned["order_id"] == "o2", "order_value"].iloc[0])
    assert pd.isna(cleaned.loc[cleaned["order_id"] == "o3", "order_value"].iloc[0])
    assert pd.isna(cleaned.loc[cleaned["order_id"] == "o3", "num_items_ordered"].iloc[0])
    assert pd.isna(cleaned.loc[cleaned["order_id"] == "o4", "num_items_ordered"].iloc[0])
    
    # DI-38: refund_value > order_value preserved raw
    assert cleaned.loc[cleaned["order_id"] == "o1", "refund_value"].iloc[0] == 50.0
    assert cleaned["refund_value"].dtype == "float64"
    assert pd.api.types.is_datetime64_any_dtype(cleaned["order_date"])


def test_clean_labels_end_to_end():
    raw_labels = pd.DataFrame({
        "country_code": ["my", "PH", "PK", "BD", "invalid_cc", "MY"],
        "order_id": ["o1", "o2", "o3", "o4", "o5", "o1"],
        "customer_id": ["c1", "c2", "c3", "", "c5", "c1"],  # o4 has empty customer_id -> dropped
        "is_fraud": ["0", "1", "invalid_target", "0", "1", "1"]  # o3 has invalid target -> dropped
    })

    cleaned, report = clean_labels(raw_labels)
    
    # Dropped:
    # - o4: empty customer_id
    # - o3: invalid_target
    # - o5: invalid_cc
    # Key duplicate: ('MY', 'o1') row 0 vs row 5 -> keep last (row 5 with is_fraud=1)
    assert len(cleaned) == 2
    assert set(cleaned["order_id"]) == {"o1", "o2"}
    assert cleaned.loc[cleaned["order_id"] == "o1", "is_fraud"].iloc[0] == 1
    assert cleaned["is_fraud"].dtype == "int64"


# ==============================================================================
# 4. IDEMPOTENCY & IMMUTABILITY TESTS
# ==============================================================================

def test_immutability():
    raw_customers = pd.DataFrame({
        "country_code": ["my", "PH"],
        "customer_id": ["c1", "c2"],
        "mobile_verified": ["yes", "False"],
        "num_orders_last_50days": ["2", "0"],
        "num_cancelled_orders_last_50days": ["0", "0"],
        "num_refund_orders_last_50days": ["0", "0"],
        "total_payment_last_50days": ["45.50", "0.0"],
        "num_associated_customers": ["1", "2"],
        "first_order_datetime": ["2022-01-01 10:00:00", "2023-05-10"]
    })
    raw_copy = raw_customers.copy(deep=True)
    
    cleaned, _ = clean_customers(raw_customers)
    
    # Caller's dataframe must not be mutated
    pd.testing.assert_frame_equal(raw_customers, raw_copy)


def test_idempotency_customers():
    raw_customers = pd.DataFrame({
        "country_code": ["my", "PH"],
        "customer_id": ["c1", "c2"],
        "mobile_verified": ["yes", "False"],
        "num_orders_last_50days": ["2", "0"],
        "num_cancelled_orders_last_50days": ["0", "0"],
        "num_refund_orders_last_50days": ["0", "0"],
        "total_payment_last_50days": ["45.50", "0.0"],
        "num_associated_customers": ["1", "2"],
        "first_order_datetime": ["2022-01-01 10:00:00", "2023-05-10"]
    })

    cleaned_once, rep1 = clean_customers(raw_customers)
    cleaned_twice, rep2 = clean_customers(cleaned_once)

    pd.testing.assert_frame_equal(cleaned_once, cleaned_twice)
    assert rep2["rows_in"] == rep2["rows_out"]
    assert rep2["country_code_normalized"] == 0
    assert rep2["exact_duplicates_dropped"] == 0
    assert rep2["key_duplicates_dropped"] == 0


def test_clean_batch_integration():
    c = pd.DataFrame({
        "country_code": ["MY", "PH"],
        "customer_id": ["c1", "c2"],
        "mobile_verified": ["1", "0"],
        "num_orders_last_50days": ["1", "2"],
        "num_cancelled_orders_last_50days": ["0", "0"],
        "num_refund_orders_last_50days": ["0", "0"],
        "total_payment_last_50days": ["10.0", "20.0"],
        "num_associated_customers": ["1", "1"],
        "first_order_datetime": ["2023-01-01", "2023-02-01"]
    })
    o = pd.DataFrame({
        "country_code": ["MY", "PH"],
        "order_id": ["o1", "o2"],
        "collect_type": ["delivery", "pickup"],
        "payment_method": ["credit card", "balance"],
        "order_value": ["10.0", "20.0"],
        "num_items_ordered": ["1", "2"],
        "refund_value": ["0.0", "0.0"],
        "order_date": ["2023-01-01", "2023-02-01"]
    })
    l = pd.DataFrame({
        "country_code": ["MY", "PH"],
        "order_id": ["o1", "o2"],
        "customer_id": ["c1", "c2"],
        "is_fraud": ["0", "1"]
    })

    res = clean_batch(c, o, l)
    assert isinstance(res, CleanResult)
    assert len(res.customers) == 2
    assert len(res.orders) == 2
    assert len(res.labels) == 2
    assert res.report["join_coverage"]["labels_to_orders_orphan"] == 0
    assert res.report["join_coverage"]["labels_to_customers_orphan"] == 0
