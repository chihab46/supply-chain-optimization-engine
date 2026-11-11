"""Isolate live PostgreSQL tests in a disposable schema, not in production tables."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from supply_chain_opt_engine.database import initialize_schema


@pytest.fixture
def engine():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL integration tests.")
    schema = "ingestion_test_" + uuid.uuid4().hex
    admin = create_engine(url)
    with admin.begin() as connection:
        connection.exec_driver_sql(f"CREATE SCHEMA {schema}")
    scoped = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    try:
        initialize_schema(scoped, Path("sql/01_schema.sql"))
        yield scoped
    finally:
        scoped.dispose()
        with admin.begin() as connection:
            connection.exec_driver_sql(f"DROP SCHEMA {schema} CASCADE")
        admin.dispose()
