"""Transactional PostgreSQL COPY staging and repeatable scope loading."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from supply_chain_opt_engine.m5 import ScopedDataset

TABLE_KEYS = {
    "dim_calendar": ["date"],
    "dim_sku_economics": ["item_id"],
    "dim_prices": ["store_id", "item_id", "wm_yr_wk"],
    "fact_daily_sales": ["store_id", "item_id", "date"],
}


def initialize_schema(engine: Engine, schema_path: Path) -> None:
    """Run the repository bootstrap against PostgreSQL."""
    if engine.dialect.name != "postgresql":
        raise ValueError("Ingestion requires PostgreSQL, not an emulated SQL dialect.")
    with engine.begin() as connection:
        cursor = connection.connection.driver_connection.cursor()
        try:
            # No parameters: use the driver's simple-query path for this trusted
            # multi-statement bootstrap with both supported PostgreSQL drivers.
            cursor.execute(schema_path.read_text())
        finally:
            cursor.close()


def load_dataset(engine: Engine, dataset: ScopedDataset, economics: pd.DataFrame) -> dict[str, int]:
    """Load four tables atomically. Exact reruns upsert the same keys.

    Reject scopes that would leave old fact rows outside the new panel. COPY
    supports psycopg 3 and OS-provided psycopg2 connections.
    """
    if engine.dialect.name != "postgresql":
        raise ValueError("Ingestion requires PostgreSQL.")
    tables = {
        "dim_calendar": dataset.calendar,
        "dim_sku_economics": economics,
        "dim_prices": dataset.prices,
        "fact_daily_sales": dataset.sales,
    }
    with engine.begin() as connection:
        connection.exec_driver_sql("SELECT pg_advisory_xact_lock(731082604)")
        cursor = connection.connection.driver_connection.cursor()
        try:
            for name, frame in tables.items():
                columns = ", ".join(f'"{column}"' for column in frame.columns)
                staging = f"stage_{name}"
                connection.exec_driver_sql(
                    f"CREATE TEMP TABLE {staging} (LIKE {name} INCLUDING ALL) ON COMMIT DROP"
                )
                statement = f"COPY {staging} ({columns}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"
                if hasattr(cursor, "copy"):
                    with cursor.copy(statement) as copy:
                        for offset in range(0, len(frame), 10000):
                            block = frame.iloc[offset : offset + 10000].to_csv(
                                index=False,
                                header=offset == 0,
                                date_format="%Y-%m-%d",
                            )
                            copy.write(block)
                else:
                    for offset in range(0, len(frame), 10000):
                        block = frame.iloc[offset : offset + 10000].to_csv(
                            index=False,
                            date_format="%Y-%m-%d",
                        )
                        cursor.copy_expert(statement, StringIO(block))
            outside_scope = connection.exec_driver_sql("""
                SELECT EXISTS (
                    SELECT 1 FROM fact_daily_sales f
                    WHERE NOT EXISTS (
                        SELECT 1 FROM stage_fact_daily_sales s
                        WHERE (s.store_id, s.item_id, s.date) = (f.store_id, f.item_id, f.date)
                    )
                )
            """).scalar_one()
            if outside_scope:
                raise ValueError(
                    "Database contains facts outside this scope. "
                    "Use a fresh database for a different scope; existing data was not changed."
                )
            mismatch = connection.exec_driver_sql("""
                SELECT EXISTS (
                    SELECT 1 FROM dim_sku_economics e
                    JOIN stage_dim_sku_economics s USING (item_id)
                    WHERE e IS DISTINCT FROM s
                )
            """).scalar_one()
            if mismatch:
                raise ValueError(
                    "Existing SKU economics differ; use a fresh database for this run."
                )
            for name, frame in tables.items():
                columns = ", ".join(f'"{column}"' for column in frame.columns)
                keys = ", ".join(TABLE_KEYS[name])
                updates = ", ".join(
                    f'"{column}" = EXCLUDED."{column}"'
                    for column in frame.columns
                    if column not in TABLE_KEYS[name]
                )
                connection.exec_driver_sql(
                    f"INSERT INTO {name} ({columns}) SELECT {columns} FROM stage_{name} "
                    f"ON CONFLICT ({keys}) DO UPDATE SET {updates}"
                )
            connection.exec_driver_sql(
                "ALTER TABLE fact_daily_sales ALTER COLUMN is_available SET NOT NULL"
            )
        finally:
            cursor.close()
    return {name: len(frame) for name, frame in tables.items()}
