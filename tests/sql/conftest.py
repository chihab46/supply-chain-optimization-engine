"""Execute feature SQL against real PostgreSQL without requiring server sockets."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from datetime import date, timedelta
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def single_user_postgres(tmp_path_factory):
    postgres = shutil.which("postgres")
    if not postgres:
        candidates = sorted(Path("/usr/lib/postgresql").glob("*/bin/postgres"))
        postgres = str(candidates[-1]) if candidates else None
    if not postgres or (hasattr(os, "geteuid") and os.geteuid() == 0):
        pytest.skip("SQL tests require local PostgreSQL server binaries and a non-root user.")
    initdb = Path(postgres).parent / "initdb"
    cluster = tmp_path_factory.mktemp("features-postgres") / "cluster"
    result = subprocess.run(
        [str(initdb), "-D", str(cluster), "-A", "trust", "--no-locale", "--encoding=UTF8"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        pytest.fail(f"Failed to initialize local PostgreSQL: {result.stderr}")

    def run(script, expected_error=None):
        # Single-user input is newline-delimited. All comments in our bootstrap
        # scripts occupy their own lines. Keep each transaction in one input line.
        sql = " ".join(line for line in script.splitlines() if not line.lstrip().startswith("--"))
        result = subprocess.run(
            [postgres, "--single", "-D", str(cluster), "postgres"],
            input=sql + "\n",
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        if expected_error:
            assert expected_error in result.stderr, result.stderr
        else:
            assert "ERROR:" not in result.stderr, result.stderr
            assert "FEATURE_TEST_EXECUTED" in result.stdout, result.stdout
        return result

    return run


@pytest.fixture
def feature_sql_case(tmp_path, single_user_postgres):
    start = date(2019, 1, 1)
    origin_day = 392
    origin = start + timedelta(days=origin_day - 1)
    calendar = []
    for day in range(1, 449):
        current = start + timedelta(days=day - 1)
        holiday = day == origin_day + 14
        second_holiday = day == origin_day + 3
        cultural = day == origin_day + 10
        calendar.append(
            {
                "date": current,
                "d": day,
                "wm_yr_wk": 1000 + (day - 1) // 7,
                "weekday": current.strftime("%A"),
                "wday": current.isoweekday(),
                "month": current.month,
                "year": current.year,
                "event_name_1": "Holiday" if holiday else ("Festival" if cultural else None),
                "event_type_1": "National" if holiday else ("Cultural" if cultural else None),
                "event_name_2": "SecondSlotHoliday" if second_holiday else None,
                "event_type_2": "National" if second_holiday else None,
                "snap_ca": day % 4 == 0,
            }
        )
    economics = [
        {
            "item_id": "FOODS_1_001",
            "dept_id": "FOODS_1",
            "cat_id": "FOODS",
            "unit_purchase_cost": 2,
            "daily_holding_cost": 0.01,
            "stockout_penalty": 0.05,
            "unit_volume_m3": 0.001,
            "lead_time_days": 2,
            "minimum_order_quantity": 6,
            "fixed_order_cost": 2,
        }
    ]
    prices, sales = [], []
    for store, offset in [("CA_1", 0), ("CA_2", 100)]:
        prices.extend(
            {
                "store_id": store,
                "item_id": "FOODS_1_001",
                "wm_yr_wk": 1000 + week,
                "sell_price": 10 + week,
            }
            for week in range(64)
        )
        sales.extend(
            {
                "store_id": store,
                "item_id": "FOODS_1_001",
                "date": start + timedelta(days=day - 1),
                "sales": offset + day % 5,
                "is_available": True,
            }
            for day in range(origin_day - 83, origin_day + 29)
        )
    tables = {
        "dim_calendar": calendar,
        "dim_sku_economics": economics,
        "dim_prices": prices,
        "fact_daily_sales": sales,
    }
    copies = []
    for table, rows in tables.items():
        path = tmp_path / f"{table}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        copies.append(f"COPY {table} FROM '{path}' WITH (FORMAT CSV, HEADER TRUE);")
    setup = (
        "BEGIN; CREATE SCHEMA features_test; SET search_path TO features_test;\n"
        + Path("sql/01_schema.sql").read_text()
        + "\n"
        + "\n".join(copies)
        + "\n"
        + Path("sql/02_feature_view.sql").read_text()
    )

    def run(check, expected_error=None):
        return single_user_postgres(
            setup + "\n" + check + "\nSELECT 'FEATURE_TEST_EXECUTED'; ROLLBACK;", expected_error
        )

    return run, origin, origin_day
