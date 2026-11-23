"""Database settings for every script that talks to RDS (PBI-09.4).

Settings come from environment variables, or from a `.env` file in the repo root
(real environment variables win). Nothing is hard-coded, so IT managers can change
the database without editing code:

    DB_HOST      RDS endpoint, e.g. spbuy-db.xxxxxxxx.us-east-1.rds.amazonaws.com  (required)
    DB_PORT      default 3306
    DB_USER      database user                                                    (required)
    DB_PASSWORD  its password                                                     (required)
    DB_NAME      default spbuy
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pymysql

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED = ("DB_HOST", "DB_USER", "DB_PASSWORD")


class ConfigError(RuntimeError):
    """Raised when required database settings are missing or invalid."""


@dataclass(frozen=True)
class DBSettings:
    host: str
    port: int
    user: str
    password: str
    database: str

    def __repr__(self) -> str:  # never print the password
        return f"DBSettings(host={self.host!r}, port={self.port}, user={self.user!r}, database={self.database!r})"


def load_dotenv(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from .env into the environment (existing variables win)."""
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_settings() -> DBSettings:
    """Return the database settings, raising ConfigError if any required one is missing."""
    load_dotenv()
    missing = [k for k in REQUIRED if not os.environ.get(k)]
    if missing:
        raise ConfigError(
            f"Missing database settings: {', '.join(missing)}. "
            "Set them as environment variables or in .env (see .env.example)."
        )
    try:
        port = int(os.environ.get("DB_PORT", "3306"))
    except ValueError as exc:
        raise ConfigError(f"DB_PORT must be a number, got {os.environ['DB_PORT']!r}") from exc
    return DBSettings(
        host=os.environ["DB_HOST"],
        port=port,
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ.get("DB_NAME", "spbuy"),
    )


def connect(use_database: bool = True, **kwargs) -> pymysql.connections.Connection:
    """Open a connection to RDS. use_database=False connects without selecting the database."""
    s = get_settings()
    return pymysql.connect(
        host=s.host,
        port=s.port,
        user=s.user,
        password=s.password,
        database=s.database if use_database else None,
        connect_timeout=10,
        charset="utf8mb4",
        **kwargs,
    )
