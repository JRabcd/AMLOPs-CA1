# RDS schema design (PBI-05.2)

The team database on AWS RDS (MySQL 8) holds the **cleaned** data for every batch
(`base`, then each new `XXX` batch). The ETL (PBI-09) cleans each CSV batch and loads it here; retraining
(PBI-10) and the individual analytics (IND-01) read from here.

Script: [`sql/schema.sql`](../sql/schema.sql). Apply it with [`src/db/apply_schema.py`](../src/db/apply_schema.py).

## What the base data showed

Profiled from `customers_base.csv` (901,312 rows), `orders_base.csv` (949,889) and `labels_base.csv` (950,318):

| Finding | Effect on the schema |
|---|---|
| `customer_id` is unique in customers | Primary key `customer_id` |
| `order_id` repeats 3 times in orders and labels, each time in a **different country** for a **different customer** (e.g. `431b5e1fcd` in MY and PK) | These are separate orders, not duplicates. Primary key is **`(country_code, order_id)`** for both orders and labels |
| `(UPPER(country_code), order_id)` is unique in both orders and labels | The composite key holds once cleaning upper-cases country codes |
| 443 labels have no matching order; 14 labels point to customers not in customers | **No foreign keys.** Hard FKs would reject real rows, and a new batch's files may load in any order. Joins use indexed columns instead |
| Lower-case country codes (`pk`, `my`, `ph`), `mobile_verified` = `True`/`False`/`yes`, `collect_type` = `Delivery`/`delivery`/`pickup` | Fixed by cleaning (PBI-03) before load; columns store the cleaned form |
| `order_date` mixes `2023/04/09`, `2023-04-09` and `2023/Feb/09`; `first_order_datetime` is `YYYY-MM-DD HH:MM:SS` with 712 blanks | Stored as `DATE` and `DATETIME`; `first_order_datetime` allows `NULL` |
| `payment_method` blank in 21,501 orders; `num_items_ordered` blank once | Both columns allow `NULL` (cleaning decides whether to fill them) |
| Money values have long float tails (e.g. `5.387818896776103` SGD); max order 662.88, max 50-day payment 42,569.30 | `DECIMAL(12,4)`: exact, enough range, 4 decimal places |
| Counts are whole numbers; largest is 2,162 orders in 50 days | `SMALLINT UNSIGNED` (0–65,535) |
| IDs are 10-character hex strings | `VARCHAR(20)`, leaving room if a later batch uses longer IDs |

## Tables

### `customers`: one row per customer
| Column | Type | Null | Notes |
|---|---|---|---|
| `customer_id` | VARCHAR(20) | no | **PK** |
| `country_code` | CHAR(2) | no | Upper-case ISO code; indexed |
| `mobile_verified` | BOOLEAN | no | |
| `num_orders_last_50days` | SMALLINT UNSIGNED | no | |
| `num_cancelled_orders_last_50days` | SMALLINT UNSIGNED | no | |
| `num_refund_orders_last_50days` | SMALLINT UNSIGNED | no | |
| `total_payment_last_50days` | DECIMAL(12,4) | no | SGD |
| `num_associated_customers` | SMALLINT UNSIGNED | no | |
| `first_order_datetime` | DATETIME | yes | 712 blanks in base |
| `batch_id` | VARCHAR(32) | no | Batch the row last came from, e.g. `base` |
| `loaded_at` | TIMESTAMP | no | Set on insert and update |

A customer can appear again in a later batch with newer 50-day figures, so the ETL should **upsert**
customers (`INSERT … ON DUPLICATE KEY UPDATE`).

### `orders`: one row per order
| Column | Type | Null | Notes |
|---|---|---|---|
| `country_code` | CHAR(2) | no | **PK** part 1 |
| `order_id` | VARCHAR(20) | no | **PK** part 2 |
| `collect_type` | VARCHAR(20) | no | e.g. `delivery`, `pickup` after cleaning |
| `payment_method` | VARCHAR(40) | yes | |
| `order_value` | DECIMAL(12,4) | no | SGD |
| `num_items_ordered` | SMALLINT UNSIGNED | yes | |
| `refund_value` | DECIMAL(12,4) | no | SGD, default 0 |
| `order_date` | DATE | no | Indexed for time-based queries |
| `batch_id`, `loaded_at` | | no | As in customers |

### `labels`: one fraud label per order
| Column | Type | Null | Notes |
|---|---|---|---|
| `country_code` | CHAR(2) | no | **PK** part 1 |
| `order_id` | VARCHAR(20) | no | **PK** part 2 |
| `customer_id` | VARCHAR(20) | no | Indexed, for joins to customers |
| `is_fraud` | TINYINT(1) | no | 0 or 1, enforced by a CHECK constraint |
| `batch_id`, `loaded_at` | | no | As in customers |

### `etl_batches`: one row per ETL run
| Column | Type | Notes |
|---|---|---|
| `batch_id` | VARCHAR(32) **PK** | Postfix of the files, e.g. `base` |
| `customers_url`, `orders_url`, `labels_url` | VARCHAR(500) | Where the files came from |
| `rows_customers`, `rows_orders`, `rows_labels` | INT UNSIGNED | Rows loaded |
| `status` | VARCHAR(20) | `running`, `success` or `failed` |
| `started_at`, `finished_at` | DATETIME | |

This gives the ETL (PBI-09) a place to log row counts, lets it skip a batch that already loaded
successfully, and gives retraining runs (PBI-10) a batch name for their MLflow run names.

## How the tables join

```
labels (country_code, order_id) ──► orders (country_code, order_id)
labels.customer_id              ──► customers.customer_id
```

Training data = labels joined to orders on `(country_code, order_id)` and to customers on `customer_id`.
Use inner joins for training, so the orphan rows found above are left out.
