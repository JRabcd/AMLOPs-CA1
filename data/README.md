# Data

CSV files are **not committed** to git (they are large, and new batches arrive from the operations team).
Keep your local copies in this folder; `.gitignore` ignores everything here except this README.

Each batch comes as three files with the same postfix (`base` for the historical data, `XXX` for later batches):

| File | Contents | Key |
|---|---|---|
| `customers_XXX.csv` | One row per customer: country, mobile verification, activity in the last 50 days, first order time | `customer_id` |
| `orders_XXX.csv` | One row per order: collection type, payment method, value, items, refund, order date | `country_code` + `order_id` |
| `labels_XXX.csv` | Fraud label per order (`is_fraud`: 1 = fraud, 0 = not fraud) and the customer who placed it | `country_code` + `order_id` |

`order_id` alone is **not** unique: the same ID can appear in two countries for different orders,
so orders and labels are identified by `country_code` + `order_id`. See [`docs/rds-schema.md`](../docs/rds-schema.md).
