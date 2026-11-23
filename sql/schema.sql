-- SP-buy fraud detection: RDS (MySQL 8) schema
-- PBI-05.3. Design notes: docs/rds-schema.md
-- Safe to run more than once: every statement uses IF NOT EXISTS.
-- Keys are NOT NULL. Every other data column allows NULL, because cleaning (src/cleaning.py)
-- sets invalid values to NULL rather than dropping the row (see docs/data_issue_log.md).
-- Changes to tables that already exist go in sql/migrations/.

CREATE DATABASE IF NOT EXISTS spbuy
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE spbuy;

-- One row per customer. Later batches may update a customer's 50-day figures (upsert on customer_id).
CREATE TABLE IF NOT EXISTS customers (
  customer_id                       VARCHAR(20)       NOT NULL,
  country_code                      CHAR(2)           NOT NULL,
  mobile_verified                   BOOLEAN           NULL,
  num_orders_last_50days            SMALLINT UNSIGNED NULL,
  num_cancelled_orders_last_50days  SMALLINT UNSIGNED NULL,
  num_refund_orders_last_50days     SMALLINT UNSIGNED NULL,
  total_payment_last_50days         DECIMAL(12,4)     NULL,
  num_associated_customers          SMALLINT UNSIGNED NULL,
  first_order_datetime              DATETIME          NULL,
  batch_id                          VARCHAR(32)       NOT NULL,
  loaded_at                         TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (customer_id),
  KEY idx_customers_country (country_code)
) ENGINE=InnoDB;

-- One row per order. order_id alone is not unique (same ID in two countries), so the key is (country_code, order_id).
CREATE TABLE IF NOT EXISTS orders (
  country_code       CHAR(2)           NOT NULL,
  order_id           VARCHAR(20)       NOT NULL,
  collect_type       VARCHAR(20)       NULL,
  payment_method     VARCHAR(40)       NULL,
  order_value        DECIMAL(12,4)     NULL,
  num_items_ordered  SMALLINT UNSIGNED NULL,
  refund_value       DECIMAL(12,4)     NULL,
  order_date         DATE              NULL,
  batch_id           VARCHAR(32)       NOT NULL,
  loaded_at          TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (country_code, order_id),
  KEY idx_orders_date (order_date)
) ENGINE=InnoDB;

-- One fraud label per order. No foreign keys: some labels have no matching order/customer,
-- and a batch's files may load in any order. Joins use the indexed columns instead.
CREATE TABLE IF NOT EXISTS labels (
  country_code  CHAR(2)      NOT NULL,
  order_id      VARCHAR(20)  NOT NULL,
  customer_id   VARCHAR(20)  NOT NULL,
  is_fraud      TINYINT(1)   NOT NULL,
  batch_id      VARCHAR(32)  NOT NULL,
  loaded_at     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (country_code, order_id),
  KEY idx_labels_customer (customer_id),
  CONSTRAINT chk_labels_is_fraud CHECK (is_fraud IN (0, 1))
) ENGINE=InnoDB;

-- One row per ETL run: source files, row counts and outcome (used by PBI-09 and for MLflow run names in PBI-10).
CREATE TABLE IF NOT EXISTS etl_batches (
  batch_id        VARCHAR(32)   NOT NULL,
  customers_url   VARCHAR(500)  NULL,
  orders_url      VARCHAR(500)  NULL,
  labels_url      VARCHAR(500)  NULL,
  rows_customers  INT UNSIGNED  NULL,
  rows_orders     INT UNSIGNED  NULL,
  rows_labels     INT UNSIGNED  NULL,
  status          VARCHAR(20)   NOT NULL DEFAULT 'running',
  started_at      DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  finished_at     DATETIME      NULL,
  error_message   VARCHAR(500)  NULL,
  PRIMARY KEY (batch_id),
  CONSTRAINT chk_etl_batches_status CHECK (status IN ('running', 'success', 'failed'))
) ENGINE=InnoDB;
