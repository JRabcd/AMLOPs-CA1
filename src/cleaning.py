"""
ST1519 Agile MLOps CA1 - PBI-03: Reusable Data-Cleaning Module
Owner: Jian Rong (Group 3)

Provides deterministic, idempotent, and vectorized data cleaning functions
for customers, orders, and labels datasets. Absorbs EDA findings from PBI-02
and satisfies RDS schema specifications (PBI-05).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger("cleaning")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [cleaning]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ==============================================================================
# CONFIGURATION BLOCK (Source of truth derived from PBI-02 EDA & RDS Schema)
# ==============================================================================

VALID_COUNTRIES: set[str] = {"BD", "MY", "PH", "PK", "TH"}

TRUE_TOKENS: set[str] = {"true", "t", "1", "1.0", "yes", "y"}
FALSE_TOKENS: set[str] = {"false", "f", "0", "0.0", "no", "n"}

# Explicit date formats in descending order of specificity
DATE_FORMATS: list[str] = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y/%b/%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
]

# Canonical category synonym mappings (normalized string -> canonical value)
PAYMENT_METHOD_SYNONYMS: dict[str, str] = {
    "on delivery": "payment on delivery",
    "payment on delivery": "payment on delivery",
    "online banking": "online banking",
    "cybersource": "cybersource",
    "credit card": "credit card",
    "antfinancial gcash": "antfinancial gcash",
    "generic creditcard": "generic creditcard",
    "balance": "balance",
    "antfinancial bkash": "antfinancial bkash",
    "antfinancial truemoney": "antfinancial truemoney",
    "boost wallet": "boost wallet",
    "jazzcash wallet": "jazzcash wallet",
    "invoice": "invoice",
    "adyen hpp molpay": "adyen hpp molpay",
    "no payment": "no payment",
    "cybersource applepay": "cybersource applepay",
    "paypal": "paypal",
    "xendit directdebit": "xendit directdebit",
}

COLLECT_TYPE_SYNONYMS: dict[str, str] = {
    "delivery": "delivery",
    "pickup": "pickup",
}

PLACEHOLDER_NULL_TOKENS: set[str] = {
    "",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
    "<na>",
    "undefined",
    "?",
}

EXPECTED_COLUMNS: dict[str, list[str]] = {
    "customers": [
        "country_code",
        "customer_id",
        "mobile_verified",
        "num_orders_last_50days",
        "num_cancelled_orders_last_50days",
        "num_refund_orders_last_50days",
        "total_payment_last_50days",
        "num_associated_customers",
        "first_order_datetime",
    ],
    "orders": [
        "country_code",
        "order_id",
        "collect_type",
        "payment_method",
        "order_value",
        "num_items_ordered",
        "refund_value",
        "order_date",
    ],
    "labels": [
        "country_code",
        "order_id",
        "customer_id",
        "is_fraud",
    ],
}


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class CleanResult:
    """Dataclass holding cleaned DataFrames and comprehensive pipeline report."""
    customers: pd.DataFrame
    orders: pd.DataFrame
    labels: pd.DataFrame
    report: dict[str, Any]


# ==============================================================================
# SCHEMA VALIDATION
# ==============================================================================

def validate_schema(df: pd.DataFrame, table: str) -> None:
    """
    Validate that df contains exactly the expected columns for the table.
    Raises ValueError with descriptive differences if schema is invalid.
    """
    if table not in EXPECTED_COLUMNS:
        raise ValueError(f"Unknown table '{table}'. Expected one of {list(EXPECTED_COLUMNS.keys())}")
    
    expected = set(EXPECTED_COLUMNS[table])
    actual = set(df.columns)
    
    missing = expected - actual
    extra = actual - expected
    
    if missing or extra:
        errors = []
        if missing:
            errors.append(f"missing column(s): {sorted(missing)}")
        if extra:
            errors.append(f"unexpected extra column(s): {sorted(extra)}")
        msg = f"Schema validation failed for '{table}': " + "; ".join(errors)
        logger.error(msg)
        raise ValueError(msg)


# ==============================================================================
# ATOMIC CLEANING HELPERS (Vectorized, idempotent, pure)
# ==============================================================================

def canon(s: pd.Series) -> pd.Series:
    """
    Lowercase, strip whitespace, and collapse punctuation/whitespace into single spaces.
    Absorbed directly from PBI-02 EDA helper.
    """
    return s.astype(str).str.strip().str.lower().str.replace(r"[\s_\-]+", " ", regex=True)


def normalize_country_code(s: pd.Series) -> tuple[pd.Series, int]:
    """
    Strip and upper-case country codes.
    Returns (cleaned_series, count_modified).
    """
    # Work on string representation while preserving missing values
    is_na = s.isna() | s.astype(str).str.strip().str.lower().isin(PLACEHOLDER_NULL_TOKENS)
    cleaned = s.astype(str).str.strip().str.upper()
    cleaned = cleaned.mask(is_na, pd.NA)
    
    # Modified count compares non-null raw with cleaned
    modified_count = int((s.notna() & ~is_na & (s.astype(str) != cleaned)).sum())
    return cleaned, modified_count


def to_boolean(s: pd.Series) -> tuple[pd.Series, int, int]:
    """
    Map boolean-like tokens to pandas nullable boolean dtype.
    Returns (cleaned_series, count_non_literal_fixed, count_unmappable).
    """
    raw_str = s.astype(str).str.strip().str.lower()
    is_na = s.isna() | raw_str.isin(PLACEHOLDER_NULL_TOKENS)
    
    # Vectorized distinct-mapping for O(distinct) speed
    distinct_vals = pd.Series(s.dropna().unique())
    token_map: dict[Any, Any] = {}
    for val in distinct_vals:
        norm = str(val).strip().lower()
        if norm in PLACEHOLDER_NULL_TOKENS:
            token_map[val] = pd.NA
        elif norm in TRUE_TOKENS:
            token_map[val] = True
        elif norm in FALSE_TOKENS:
            token_map[val] = False
        else:
            token_map[val] = pd.NA
            
    mapped = s.map(token_map).astype("boolean")
    
    # Count tokens that were not strictly literal 'True' / 'False' but mapped successfully
    literal_mask = s.astype(str).isin(["True", "False"])
    valid_mapped = mapped.notna()
    fixed_count = int((~literal_mask & valid_mapped & ~is_na).sum())
    
    # Count unmappable non-null values
    unmappable_count = int((~is_na & mapped.isna()).sum())
    
    return mapped, fixed_count, unmappable_count


def normalize_category(
    s: pd.Series,
    synonyms: dict[str, str] | None = None,
    allow_null: bool = True,
) -> tuple[pd.Series, int]:
    """
    Normalize categorical string column:
    - strip, lowercase, collapse spaces
    - map synonyms to canonical form
    - placeholder/null tokens mapped to None/pd.NA per RDS schema
    Returns (cleaned_series, count_fixed).
    """
    distinct_raw = pd.Series(s.dropna().unique())
    val_map: dict[Any, Any] = {}
    
    for raw_val in distinct_raw:
        c = re.sub(r"[\s_\-]+", " ", str(raw_val).strip().lower())
        if c in PLACEHOLDER_NULL_TOKENS:
            val_map[raw_val] = pd.NA if allow_null else "unknown"
        elif synonyms and c in synonyms:
            val_map[raw_val] = synonyms[c]
        else:
            val_map[raw_val] = c

    cleaned = s.map(val_map).astype("string")
    if allow_null:
        # Also ensure raw NA becomes pd.NA
        cleaned = cleaned.mask(s.isna(), pd.NA)
    else:
        cleaned = cleaned.fillna("unknown")

    # Fixed count: raw was not null and differed from cleaned output
    fixed_count = int((s.notna() & (s.astype(str) != cleaned.fillna("<NA>"))).sum())
    return cleaned, fixed_count


def parse_dates(
    s: pd.Series,
    formats: list[str] = DATE_FORMATS,
) -> tuple[pd.Series, dict[str, int], int]:
    """
    Parse string dates against an explicit list of known formats.
    Absorbed from PBI-02 EDA helper `parse_mixed_dates`.
    Returns (datetime_series, format_usage_dict, count_unparseable).
    """
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    used: dict[str, int] = {}
    
    # If s is already datetime64, preserve it
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.astype("datetime64[ns]"), {"already_datetime": len(s.dropna())}, 0

    raw_str = s.astype(str).str.strip()
    is_placeholder = s.isna() | raw_str.str.lower().isin(PLACEHOLDER_NULL_TOKENS)
    todo = ~is_placeholder
    
    for f in formats:
        idx = s.index[todo & out.isna()]
        if len(idx) == 0:
            break
        # Fast vectorized format-specific parse
        parsed = pd.to_datetime(raw_str.loc[idx], format=f, errors="coerce")
        ok = parsed.notna()
        out.loc[idx[ok]] = parsed[ok]
        count = int(ok.sum())
        if count > 0:
            used[f] = count

    unparseable_count = int((todo & out.isna()).sum())
    return out, used, unparseable_count


def coerce_numeric(
    s: pd.Series,
    is_integer: bool = False,
    min_val: float | None = None,
    min_inclusive: bool = True,
) -> tuple[pd.Series, int, int]:
    """
    Coerce Series to numeric dtype (Int64 for integers, float64 for floats).
    Flags non-numeric text and domain rule violations.
    Returns (cleaned_series, non_numeric_count, violation_count).
    """
    if pd.api.types.is_numeric_dtype(s):
        coerced = s.astype(float)
        non_numeric_count = 0
    else:
        raw_str = s.astype(str).str.strip()
        is_placeholder = s.isna() | raw_str.str.lower().isin(PLACEHOLDER_NULL_TOKENS)
        coerced = pd.to_numeric(raw_str, errors="coerce")
        non_numeric_count = int((~is_placeholder & coerced.isna()).sum())
    
    # Domain boundary checks
    violation_mask = pd.Series(False, index=s.index)
    if min_val is not None:
        if min_inclusive:
            violation_mask |= (coerced < min_val)
        else:
            violation_mask |= (coerced <= min_val)
            
    if is_integer:
        # Non-integer check (e.g. 1.5 in count column)
        violation_mask |= (coerced.notna() & (coerced % 1 != 0))

    violation_count = int((coerced.notna() & violation_mask).sum())
    coerced = coerced.mask(violation_mask, np.nan)
    
    if is_integer:
        out = coerced.round().astype("Int64")
    else:
        out = coerced.astype("float64")
        
    return out, non_numeric_count, violation_count


def drop_duplicates_by_key(
    df: pd.DataFrame,
    key_cols: list[str],
) -> tuple[pd.DataFrame, int, int]:
    """
    Deduplicate DataFrame:
    1. Drop exact duplicate rows.
    2. Drop composite key duplicates, keeping the last occurrence.
    Returns (deduped_df, exact_dups_count, key_dups_count).
    """
    n_in = len(df)
    df_no_exact = df.drop_duplicates()
    exact_dups = n_in - len(df_no_exact)
    
    df_deduped = df_no_exact.drop_duplicates(subset=key_cols, keep="last")
    key_dups = len(df_no_exact) - len(df_deduped)
    
    return df_deduped, exact_dups, key_dups


# ==============================================================================
# PUBLIC TABLE CLEANING APIS
# ==============================================================================

def clean_customers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clean customers DataFrame according to PBI-02 findings and RDS schema:
    - Normalizes country_code to uppercase; validates against VALID_COUNTRIES
    - Normalizes mobile_verified to boolean
    - Coerces counts to Int64, payment to float64
    - Parses first_order_datetime to datetime64[ns]
    - Drops invalid/missing keys and key duplicates
    """
    validate_schema(df, "customers")
    rows_in = len(df)
    report: dict[str, Any] = {"table": "customers", "rows_in": rows_in}
    
    df_clean = df.copy()

    # 1. Country code normalization
    cc_cleaned, cc_mod = normalize_country_code(df_clean["country_code"])
    df_clean["country_code"] = cc_cleaned.astype("string")
    report["country_code_normalized"] = cc_mod
    
    # 2. Key validation: drop missing/invalid keys
    invalid_cc = ~df_clean["country_code"].isin(VALID_COUNTRIES) | df_clean["country_code"].isna()
    cust_id_str = df_clean["customer_id"].astype(str).str.strip()
    invalid_id = df_clean["customer_id"].isna() | cust_id_str.str.lower().isin(PLACEHOLDER_NULL_TOKENS)
    bad_keys = invalid_cc | invalid_id
    invalid_keys_count = int(bad_keys.sum())
    report["invalid_keys_dropped"] = invalid_keys_count
    df_clean = df_clean[~bad_keys].copy()
    df_clean["customer_id"] = df_clean["customer_id"].astype("string")

    # 3. Deduplication on composite key (country_code, customer_id)
    df_clean, exact_dups, key_dups = drop_duplicates_by_key(df_clean, ["country_code", "customer_id"])
    report["exact_duplicates_dropped"] = exact_dups
    report["key_duplicates_dropped"] = key_dups

    # 4. mobile_verified
    mv_clean, mv_fixed, mv_unmappable = to_boolean(df_clean["mobile_verified"])
    df_clean["mobile_verified"] = mv_clean
    report["mobile_verified_tokens_fixed"] = mv_fixed
    report["mobile_verified_unmappable_null"] = mv_unmappable

    # 5. Numeric columns (Int64 counts >= 0, float64 payment >= 0)
    int_cols = [
        "num_orders_last_50days",
        "num_cancelled_orders_last_50days",
        "num_refund_orders_last_50days",
        "num_associated_customers",
    ]
    for col in int_cols:
        col_clean, non_num, viol = coerce_numeric(df_clean[col], is_integer=True, min_val=0, min_inclusive=True)
        df_clean[col] = col_clean
        report[f"{col}_non_numeric"] = non_num
        report[f"{col}_negative_set_nan"] = viol

    tot_pay, non_num_pay, viol_pay = coerce_numeric(df_clean["total_payment_last_50days"], is_integer=False, min_val=0, min_inclusive=True)
    df_clean["total_payment_last_50days"] = tot_pay
    report["total_payment_last_50days_non_numeric"] = non_num_pay
    report["total_payment_last_50days_negative_set_nan"] = viol_pay

    # 6. Dates
    dt_clean, used_fmts, unparseable_dt = parse_dates(df_clean["first_order_datetime"])
    df_clean["first_order_datetime"] = dt_clean
    report["first_order_datetime_formats"] = used_fmts
    report["first_order_datetime_unparseable"] = unparseable_dt

    rows_out = len(df_clean)
    report["rows_out"] = rows_out
    logger.info(f"clean_customers complete: {rows_in:,} in -> {rows_out:,} out")
    return df_clean, report


def clean_orders(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clean orders DataFrame according to PBI-02 findings and RDS schema:
    - Normalizes country_code to uppercase; validates against VALID_COUNTRIES
    - Normalizes collect_type and payment_method categories & synonyms
    - Coerces order_value > 0 (float64) and num_items_ordered >= 1 (Int64)
    - Keeps refund_value raw (float64) per PR #112 / DI-38
    - Parses order_date to datetime64[ns]
    - Drops invalid/missing keys and key duplicates
    """
    validate_schema(df, "orders")
    rows_in = len(df)
    report: dict[str, Any] = {"table": "orders", "rows_in": rows_in}
    
    df_clean = df.copy()

    # 1. Country code normalization
    cc_cleaned, cc_mod = normalize_country_code(df_clean["country_code"])
    df_clean["country_code"] = cc_cleaned.astype("string")
    report["country_code_normalized"] = cc_mod

    # 2. Key validation: drop missing/invalid keys
    invalid_cc = ~df_clean["country_code"].isin(VALID_COUNTRIES) | df_clean["country_code"].isna()
    order_id_str = df_clean["order_id"].astype(str).str.strip()
    invalid_id = df_clean["order_id"].isna() | order_id_str.str.lower().isin(PLACEHOLDER_NULL_TOKENS)
    bad_keys = invalid_cc | invalid_id
    invalid_keys_count = int(bad_keys.sum())
    report["invalid_keys_dropped"] = invalid_keys_count
    df_clean = df_clean[~bad_keys].copy()
    df_clean["order_id"] = df_clean["order_id"].astype("string")

    # 3. Deduplication on composite key (country_code, order_id)
    df_clean, exact_dups, key_dups = drop_duplicates_by_key(df_clean, ["country_code", "order_id"])
    report["exact_duplicates_dropped"] = exact_dups
    report["key_duplicates_dropped"] = key_dups

    # 4. Categoricals
    collect_clean, collect_fixed = normalize_category(
        df_clean["collect_type"], synonyms=COLLECT_TYPE_SYNONYMS, allow_null=False
    )
    df_clean["collect_type"] = collect_clean
    report["collect_type_normalized"] = collect_fixed

    pm_clean, pm_fixed = normalize_category(
        df_clean["payment_method"], synonyms=PAYMENT_METHOD_SYNONYMS, allow_null=True
    )
    df_clean["payment_method"] = pm_clean
    report["payment_method_normalized"] = pm_fixed

    # 5. Numerics
    # order_value > 0 (DI-33: non-positive set to NaN)
    ov_clean, ov_non_num, ov_viol = coerce_numeric(
        df_clean["order_value"], is_integer=False, min_val=0, min_inclusive=False
    )
    df_clean["order_value"] = ov_clean
    report["order_value_non_numeric"] = ov_non_num
    report["order_value_non_positive_set_nan"] = ov_viol

    # num_items_ordered >= 1 (DI-37: < 1 set to pd.NA)
    items_clean, items_non_num, items_viol = coerce_numeric(
        df_clean["num_items_ordered"], is_integer=True, min_val=1, min_inclusive=True
    )
    df_clean["num_items_ordered"] = items_clean
    report["num_items_ordered_non_numeric"] = items_non_num
    report["num_items_ordered_invalid_set_nan"] = items_viol

    # refund_value >= 0 (float64, keep raw per PR #112 review / DI-38)
    rv_clean, rv_non_num, rv_viol = coerce_numeric(
        df_clean["refund_value"], is_integer=False, min_val=0, min_inclusive=True
    )
    df_clean["refund_value"] = rv_clean
    report["refund_value_non_numeric"] = rv_non_num
    report["refund_value_negative_set_nan"] = rv_viol

    # 6. Dates
    dt_clean, used_fmts, unparseable_dt = parse_dates(df_clean["order_date"])
    df_clean["order_date"] = dt_clean
    report["order_date_formats"] = used_fmts
    report["order_date_unparseable"] = unparseable_dt

    rows_out = len(df_clean)
    report["rows_out"] = rows_out
    logger.info(f"clean_orders complete: {rows_in:,} in -> {rows_out:,} out")
    return df_clean, report


def clean_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clean labels DataFrame according to PBI-02 findings and RDS schema:
    - Normalizes country_code to uppercase; validates against VALID_COUNTRIES
    - Validates composite key (country_code, order_id) and customer_id
    - Validates target is_fraud in {0, 1}; drops invalid/missing target rows
    - Deduplicates on composite key (country_code, order_id)
    """
    validate_schema(df, "labels")
    rows_in = len(df)
    report: dict[str, Any] = {"table": "labels", "rows_in": rows_in}
    
    df_clean = df.copy()

    # 1. Country code normalization
    cc_cleaned, cc_mod = normalize_country_code(df_clean["country_code"])
    df_clean["country_code"] = cc_cleaned.astype("string")
    report["country_code_normalized"] = cc_mod

    # 2. Key validation: country_code, order_id, customer_id
    invalid_cc = ~df_clean["country_code"].isin(VALID_COUNTRIES) | df_clean["country_code"].isna()
    order_id_str = df_clean["order_id"].astype(str).str.strip()
    invalid_oid = df_clean["order_id"].isna() | order_id_str.str.lower().isin(PLACEHOLDER_NULL_TOKENS)
    cust_id_str = df_clean["customer_id"].astype(str).str.strip()
    invalid_cid = df_clean["customer_id"].isna() | cust_id_str.str.lower().isin(PLACEHOLDER_NULL_TOKENS)
    bad_keys = invalid_cc | invalid_oid | invalid_cid
    invalid_keys_count = int(bad_keys.sum())
    report["invalid_keys_dropped"] = invalid_keys_count
    df_clean = df_clean[~bad_keys].copy()
    df_clean["order_id"] = df_clean["order_id"].astype("string")
    df_clean["customer_id"] = df_clean["customer_id"].astype("string")

    # 3. Target is_fraud validation (must be 0 or 1)
    raw_target = pd.to_numeric(df_clean["is_fraud"], errors="coerce")
    bad_target = ~raw_target.isin([0, 1]) | raw_target.isna()
    bad_target_count = int(bad_target.sum())
    report["invalid_target_dropped"] = bad_target_count
    if bad_target_count > 0:
        logger.warning(f"Dropping {bad_target_count} rows with invalid is_fraud target")
        df_clean = df_clean[~bad_target].copy()
        raw_target = raw_target[~bad_target]
        
    df_clean["is_fraud"] = raw_target.astype("int64")

    # 4. Deduplication on composite key (country_code, order_id)
    df_clean, exact_dups, key_dups = drop_duplicates_by_key(df_clean, ["country_code", "order_id"])
    report["exact_duplicates_dropped"] = exact_dups
    report["key_duplicates_dropped"] = key_dups

    rows_out = len(df_clean)
    report["rows_out"] = rows_out
    logger.info(f"clean_labels complete: {rows_in:,} in -> {rows_out:,} out")
    return df_clean, report


def clean_batch(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    labels: pd.DataFrame,
) -> CleanResult:
    """
    Clean all three tables in a batch, compute cross-table join coverage diagnostics,
    and return CleanResult(customers, orders, labels, report).
    """
    logger.info("Starting clean_batch pipeline execution")
    c_clean, c_rep = clean_customers(customers)
    o_clean, o_rep = clean_orders(orders)
    l_clean, l_rep = clean_labels(labels)

    # Cross-table join diagnostics (informational for ETL & downstream transform.py)
    # 1. labels -> orders on (country_code, order_id)
    l_ord_keys = set(l_clean["country_code"] + "|" + l_clean["order_id"])
    o_ord_keys = set(o_clean["country_code"] + "|" + o_clean["order_id"])
    labels_orders_matched = len(l_ord_keys & o_ord_keys)
    labels_orders_orphan = len(l_ord_keys - o_ord_keys)

    # 2. labels -> customers on (country_code, customer_id)
    l_cust_keys = set(l_clean["country_code"] + "|" + l_clean["customer_id"])
    c_cust_keys = set(c_clean["country_code"] + "|" + c_clean["customer_id"])
    labels_cust_matched = len(l_cust_keys & c_cust_keys)
    labels_cust_orphan = len(l_cust_keys - c_cust_keys)

    batch_report: dict[str, Any] = {
        "customers": c_rep,
        "orders": o_rep,
        "labels": l_rep,
        "join_coverage": {
            "labels_to_orders_matched": labels_orders_matched,
            "labels_to_orders_orphan": labels_orders_orphan,
            "labels_to_customers_matched": labels_cust_matched,
            "labels_to_customers_orphan": labels_cust_orphan,
        },
    }

    logger.info(
        f"clean_batch finished: customers={len(c_clean):,}, orders={len(o_clean):,}, "
        f"labels={len(l_clean):,}, orphan_orders={labels_orders_orphan}, orphan_cust={labels_cust_orphan}"
    )

    return CleanResult(
        customers=c_clean,
        orders=o_clean,
        labels=l_clean,
        report=batch_report,
    )


def main() -> None:
    """Optional small CLI for running batch data cleaning from the command line."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Clean a batch of customers, orders, and labels CSVs."
    )
    parser.add_argument("--customers", required=True, help="Path to raw customers CSV")
    parser.add_argument("--orders", required=True, help="Path to raw orders CSV")
    parser.add_argument("--labels", required=True, help="Path to raw labels CSV")
    parser.add_argument("--out", required=True, help="Output directory for cleaned CSVs and report")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading CSVs: {args.customers}, {args.orders}, {args.labels}")
    c_df = pd.read_csv(args.customers, dtype=str)
    o_df = pd.read_csv(args.orders, dtype=str)
    l_df = pd.read_csv(args.labels, dtype=str)

    result = clean_batch(c_df, o_df, l_df)

    c_out = out_dir / "customers_clean.csv"
    o_out = out_dir / "orders_clean.csv"
    l_out = out_dir / "labels_clean.csv"
    rep_out = out_dir / "cleaning_report.json"

    logger.info(f"Writing cleaned outputs to {out_dir}")
    result.customers.to_csv(c_out, index=False)
    result.orders.to_csv(o_out, index=False)
    result.labels.to_csv(l_out, index=False)

    with open(rep_out, "w", encoding="utf-8") as f:
        json.dump(result.report, f, indent=2, default=str)

    logger.info(f"Batch cleaning finished successfully. Report saved to {rep_out}")


if __name__ == "__main__":
    main()

