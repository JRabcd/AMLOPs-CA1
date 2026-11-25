"""Unit tests for the connection check (PBI-05.5). No database or network needed."""
import pymysql
import pytest

from src.db import check_connection as cc
from src.db import config


class FakeConn:
    def __init__(self, counts):
        self.counts, self.closed, self.result = counts, False, []

    def get_server_info(self):
        return "8.4.11"

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql):
        assert sql.startswith(("SHOW", "SELECT"))  # the check must only read
        if sql == "SHOW TABLES":
            self.result = [(t,) for t in self.counts]
        else:
            self.result = [(self.counts[sql.split("`")[1]],)]

    def fetchall(self):
        return self.result

    def fetchone(self):
        return self.result[0]

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def settings(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)  # no .env here
    for key, value in dict(DB_HOST="h", DB_USER="u", DB_PASSWORD="secret", DB_PORT="3306", DB_NAME="spbuy").items():
        monkeypatch.setenv(key, value)


def test_ready_database(monkeypatch, capsys):
    conn = FakeConn({t: 1234 for t in cc.TABLES})
    monkeypatch.setattr(cc, "connect", lambda: conn)
    assert cc.main() == 0
    out = capsys.readouterr().out
    assert "1,234 rows" in out and "OK" in out and "secret" not in out
    assert conn.closed


def test_missing_table_fails(monkeypatch, capsys):
    conn = FakeConn({"customers": 1, "orders": 1, "labels": 1})
    monkeypatch.setattr(cc, "connect", lambda: conn)
    assert cc.main() == 1
    captured = capsys.readouterr()
    assert "MISSING" in captured.out and "apply_schema" in captured.err
    assert conn.closed


@pytest.mark.parametrize("code, expected", [(1045, "password"), (2003, "AWS lab"), (1049, "apply_schema"), (9999, "Unexpected")])
def test_connection_errors_give_advice(monkeypatch, capsys, code, expected):
    def fail():
        raise pymysql.err.OperationalError(code, "server message")

    monkeypatch.setattr(cc, "connect", fail)
    assert cc.main() == 1
    captured = capsys.readouterr()
    assert expected in captured.err and "secret" not in captured.out + captured.err


def test_missing_settings(monkeypatch, capsys):
    monkeypatch.delenv("DB_PASSWORD")
    assert cc.main() == 1
    assert "DB_PASSWORD" in capsys.readouterr().err
