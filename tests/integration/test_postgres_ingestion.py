"""Set TEST_DATABASE_URL to a disposable PostgreSQL DB with schema privileges."""

from __future__ import annotations

from dataclasses import replace

import pytest
from sqlalchemy.exc import IntegrityError

from supply_chain_opt_engine.database import load_dataset
from supply_chain_opt_engine.m5 import prepare_dataset
from supply_chain_opt_engine.metadata import generate_economics

pytestmark = pytest.mark.integration


def test_copy_upsert_is_repeatable_and_matches_source_counts(engine, m5_raw, scope):
    dataset = prepare_dataset(m5_raw, scope)
    economics = generate_economics(dataset)
    expected = load_dataset(engine, dataset, economics)
    assert load_dataset(engine, dataset, economics) == expected
    with engine.connect() as connection:
        for table, rows in expected.items():
            assert connection.exec_driver_sql(f"SELECT count(*) FROM {table}").scalar_one() == rows
        assert connection.exec_driver_sql(
            "SELECT sum(sales) FROM fact_daily_sales"
        ).scalar_one() == int(dataset.sales["sales"].sum())


def test_late_foreign_key_failure_rolls_back_prior_table_updates(engine, m5_raw, scope):
    dataset = prepare_dataset(m5_raw, scope)
    economics = generate_economics(dataset)
    load_dataset(engine, dataset, economics)
    calendar = dataset.calendar.copy()
    calendar["event_name_1"] = "changed"
    prices = dataset.prices.copy()
    prices["item_id"] = "nonexistent"
    # Reduce to one item to avoid duplicate staged price keys masking the FK failure.
    prices = prices.drop_duplicates(["store_id", "item_id", "wm_yr_wk"])
    with pytest.raises(IntegrityError):
        load_dataset(engine, replace(dataset, calendar=calendar, prices=prices), economics)
    with engine.connect() as connection:
        assert (
            connection.exec_driver_sql(
                "SELECT count(*) FROM dim_calendar WHERE event_name_1 = 'changed'"
            ).scalar_one()
            == 0
        )
        assert connection.exec_driver_sql(
            "SELECT count(*) FROM fact_daily_sales"
        ).scalar_one() == len(dataset.sales)


def test_changed_scope_and_economics_do_not_silently_overwrite(engine, m5_raw, scope):
    dataset = prepare_dataset(m5_raw, scope)
    economics = generate_economics(dataset)
    load_dataset(engine, dataset, economics)
    with pytest.raises(ValueError, match="outside this scope"):
        load_dataset(engine, replace(dataset, sales=dataset.sales.iloc[:-1]), economics)
    changed = economics.copy()
    changed["unit_purchase_cost"] = changed["unit_purchase_cost"] + 1
    with pytest.raises(ValueError, match="economics differ"):
        load_dataset(engine, dataset, changed)
