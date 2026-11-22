"""Create the SP-buy tables on the team RDS database (PBI-05.3).

Connection settings come from src/db/config.py (environment variables or .env).

Usage (from the repo root):

    python -m src.db.apply_schema

Safe to run more than once: sql/schema.sql only creates what does not exist yet, then every
file in sql/migrations/ is applied in name order (each is written to be safe to re-run), so a
database created from an older schema is brought up to date.
"""
import sys
from pathlib import Path

from src.db.config import ConfigError, connect, get_settings

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"
SCHEMA_FILE = SQL_DIR / "schema.sql"
MIGRATIONS_DIR = SQL_DIR / "migrations"


def statements(sql: str) -> list[str]:
    """Split the schema file into statements, dropping comment-only lines."""
    lines = [l for l in sql.splitlines() if not l.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def main() -> int:
    try:
        settings = get_settings()
        conn = connect(use_database=False, autocommit=True)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        with conn.cursor() as cur:
            for stmt in statements(SCHEMA_FILE.read_text(encoding="utf-8")):
                cur.execute(stmt)
            for migration in sorted(MIGRATIONS_DIR.glob("*.sql")):
                for stmt in statements(migration.read_text(encoding="utf-8")):
                    cur.execute(stmt)
                print(f"Applied migration {migration.name}")
            cur.execute("SHOW TABLES")
            tables = [row[0] for row in cur.fetchall()]
        print(f"Schema applied to {settings.host}. Tables in {settings.database}: {', '.join(tables)}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
