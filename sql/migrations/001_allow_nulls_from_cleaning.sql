-- Migration 001 (PBI-09): let non-key columns hold NULL, matching what src/cleaning.py outputs
-- (it sets invalid values such as order_value <= 0 or an unparseable order_date to NULL).
-- Brings tables created from the first version of sql/schema.sql in line with the current one.
-- Safe to run more than once: MODIFY COLUMN just restates the definition.

USE spbuy;

ALTER TABLE customers
  MODIFY COLUMN mobile_verified                   BOOLEAN           NULL,
  MODIFY COLUMN num_orders_last_50days            SMALLINT UNSIGNED NULL,
  MODIFY COLUMN num_cancelled_orders_last_50days  SMALLINT UNSIGNED NULL,
  MODIFY COLUMN num_refund_orders_last_50days     SMALLINT UNSIGNED NULL,
  MODIFY COLUMN total_payment_last_50days         DECIMAL(12,4)     NULL,
  MODIFY COLUMN num_associated_customers          SMALLINT UNSIGNED NULL;

ALTER TABLE orders
  MODIFY COLUMN collect_type   VARCHAR(20)   NULL,
  MODIFY COLUMN order_value    DECIMAL(12,4) NULL,
  MODIFY COLUMN refund_value   DECIMAL(12,4) NULL,
  MODIFY COLUMN order_date     DATE          NULL;
