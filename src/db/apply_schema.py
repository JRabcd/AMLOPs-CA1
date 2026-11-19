"""Create the SP-buy tables on the team RDS database (PBI-05.3).

Connection details come from environment variables (or a .env file), never from code:

    DB_HOST      RDS endpoint, e.g. spbuy-db.xxxxxxxx.us-east-1.rds.amazonaws.com
    DB_PORT      default 3306
    DB_USER      master or app user
    DB_PASSWORD  its password

Usage (from the repo root):

    python src/db/apply_schema.py

Safe to run more than once: sql/schema.sql only creates what does not exist yet.
"""
import os
import sys
from pathlib import Path

import pymysql

SCHEMA_FILE = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from .env into the environment (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def statements(sql: str) -> list[str]:
    """Split the schema file into statements, dropping comment-only lines."""
    lines = [l for l in sql.splitlines() if not l.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    missing = [k for k in ("DB_HOST", "DB_USER", "DB_PASSWORD") if not os.environ.get(k)]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}", file=sys.stderr)
        return 1

    conn = pymysql.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", "3306")),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        connect_timeout=10,
        autocommit=True,
    )
    try:
        with conn.cursor() as cur:
            for stmt in statements(SCHEMA_FILE.read_text(encoding="utf-8")):
                cur.execute(stmt)
            cur.execute("SHOW TABLES")
            tables = [row[0] for row in cur.fetchall()]
        print(f"Schema applied to {os.environ['DB_HOST']}. Tables in spbuy: {', '.join(tables)}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
