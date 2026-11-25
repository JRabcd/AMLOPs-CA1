"""Check that this computer can reach the team RDS database (PBI-05.5).

Setup: copy .env.example to .env and fill it in (ask the team for the endpoint and password).

    python -m src.db.check_connection

Prints the MySQL version and the row count of each table. Exit code 0 means everything is ready,
1 means something is wrong, with a hint on what to fix. The check only reads; it changes nothing.
"""
from __future__ import annotations

import sys

import pymysql

from src.db.config import ConfigError, connect, get_settings

TABLES = ("customers", "orders", "labels", "etl_batches")

HINTS = {
    1044: "Your user has no access to this database. Check DB_USER and DB_NAME.",
    1045: "Wrong user or password. Check DB_USER and DB_PASSWORD in .env.",
    1049: "The database doesn't exist yet. Run: python -m src.db.apply_schema",
    2003: "Can't reach the server. Is the AWS lab running and the RDS instance 'Available'? Does its "
          "security group allow port 3306 from your network (some school networks block it)? Check DB_HOST.",
    2013: "The connection dropped. Check your network and try again.",
}


def hint_for(exc: Exception) -> str:
    """Plain-language advice for a MySQL error."""
    return HINTS.get(exc.args[0] if exc.args else None, "Unexpected database error.")


def count_tables(conn) -> dict[str, int | None]:
    """Rows in each expected table; None if the table doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES")
        existing = {row[0] for row in cur.fetchall()}
        counts: dict[str, int | None] = {}
        for table in TABLES:
            if table not in existing:
                counts[table] = None
                continue
            cur.execute(f"SELECT COUNT(*) FROM `{table}`")
            counts[table] = cur.fetchone()[0]
    return counts


def main() -> int:
    try:
        settings = get_settings()
        print(f"Connecting to {settings.host}:{settings.port} as {settings.user} ...", flush=True)
        conn = connect()
    except ConfigError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except pymysql.MySQLError as exc:
        print(f"FAILED: {hint_for(exc)}\n  ({exc})", file=sys.stderr)
        return 1
    try:
        print(f"Connected. MySQL {conn.get_server_info()}, database '{settings.database}'")
        counts = count_tables(conn)
    except pymysql.MySQLError as exc:
        print(f"FAILED: {hint_for(exc)}\n  ({exc})", file=sys.stderr)
        return 1
    finally:
        conn.close()
    for table, rows in counts.items():
        print(f"  {table:<12} {'MISSING' if rows is None else f'{rows:,} rows'}")
    if None in counts.values():
        print("FAILED: table(s) missing. Run: python -m src.db.apply_schema", file=sys.stderr)
        return 1
    print("OK: the database is ready to use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
