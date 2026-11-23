# ETL pipeline (PBI-09)

Loads one batch of new data (three CSVs with the same postfix) into the team RDS database:

```
read 3 CSVs (URLs or files) ──► check columns ──► clean (src/cleaning.py) ──► upsert into RDS ──► record in etl_batches
       src/etl/extract.py                                                       src/etl/load.py
```

## Run it

From the repo root, with the database settings in `.env` (see `src/db/config.py`):

```bash
# batch published as <prefix>/customers_XXX.csv, orders_XXX.csv, labels_XXX.csv
python -m src.etl.run_etl --source https://example.com/data --batch XXX

# or the three files explicitly (URLs or local paths)
python -m src.etl.run_etl --customers <url> --orders <url> --labels <url>

# read and clean only, without touching the database
python -m src.etl.run_etl --source data --batch base --dry-run
```

| Option | Effect |
|---|---|
| `--force` | Reload a batch that already loaded successfully (rows are upserted, so no duplicates) |
| `--batch-id NAME` | Record the batch under a different name than the file postfix |
| `--dry-run` | Read and clean, but don't connect to RDS |

The exit code is `0` on success (or when the batch was already loaded), and `1` on any failure.

## What it guarantees

- **Fails clearly on bad input.** A missing file, an HTTP 4xx, or a file without the expected columns stops the run with a message naming the file. Network errors are retried 3 times.
- **No duplicate keys.** Every table is written with `INSERT … ON DUPLICATE KEY UPDATE` on its primary key (`customer_id`; `(country_code, order_id)` for orders and labels). Re-running a batch updates rows instead of adding new ones.
- **Skips finished batches.** A batch whose `etl_batches.status` is `success` is skipped unless `--force` is given. A failed batch can simply be re-run.
- **Every run is recorded** in `etl_batches`: source URLs, status (`running` / `success` / `failed`), row counts of the cleaned files, start and finish time, and the error message for a failed run.
- **Row counts in the log (PBI-09.5).** For each table it logs rows read, rows kept after cleaning, and rows loaded, plus the number of orphan labels.

```
[...] [INFO] [etl]: Read orders: 5,130 rows from .../orders_devtest.csv
[...] [INFO] [etl]: Cleaned orders: 5,130 read -> 5,130 kept (0 dropped)
[...] [INFO] [etl]: Loaded orders: 5,130 rows upserted
[...] [INFO] [etl]: === ETL batch 'devtest' success in 15.1s === customers: read 5,124, cleaned 5,124, loaded 5,124 | ...
```

## Notes

- RDS stores **cleaned** data (see `docs/rds-schema.md`). Feature transformation (`src/transform.py`, PBI-04) runs when models are retrained from RDS (PBI-10), not before loading.
- Orphan labels (no matching order or customer) are loaded as they are. They're dropped by the inner join when training data is extracted.
- **Speed:** from a laptop in Singapore to RDS in us-east-1, loading runs at about 1,000–2,000 rows per second per table (≈15 s for a 5,000-row batch). The full base data (~2.8 million rows) is a one-time load of roughly half an hour. For the 3-minute demo limit (PBI-12), new batches of up to about 50,000 rows per table fit comfortably.
- Schema changes to tables that already exist go in `sql/migrations/`. `python -m src.db.apply_schema` applies the schema and then every migration.
