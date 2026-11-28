-- Migration 002 (PBI-12): index loaded_at, so retraining can fetch only rows changed since its last read
-- (src/rds_cache.py) without scanning whole tables.
-- Safe to run more than once: each index is only added if it doesn't exist yet.

USE spbuy;

SET @add_index = (SELECT IF(COUNT(*) = 0, 'ALTER TABLE customers ADD INDEX idx_customers_loaded_at (loaded_at)', 'DO 0')
  FROM information_schema.statistics
  WHERE table_schema = DATABASE() AND table_name = 'customers' AND index_name = 'idx_customers_loaded_at');
PREPARE add_index FROM @add_index;
EXECUTE add_index;
DEALLOCATE PREPARE add_index;

SET @add_index = (SELECT IF(COUNT(*) = 0, 'ALTER TABLE orders ADD INDEX idx_orders_loaded_at (loaded_at)', 'DO 0')
  FROM information_schema.statistics
  WHERE table_schema = DATABASE() AND table_name = 'orders' AND index_name = 'idx_orders_loaded_at');
PREPARE add_index FROM @add_index;
EXECUTE add_index;
DEALLOCATE PREPARE add_index;

SET @add_index = (SELECT IF(COUNT(*) = 0, 'ALTER TABLE labels ADD INDEX idx_labels_loaded_at (loaded_at)', 'DO 0')
  FROM information_schema.statistics
  WHERE table_schema = DATABASE() AND table_name = 'labels' AND index_name = 'idx_labels_loaded_at');
PREPARE add_index FROM @add_index;
EXECUTE add_index;
DEALLOCATE PREPARE add_index;
