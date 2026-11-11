"""Boundary, availability and leakage tests executed by PostgreSQL itself."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from statistics import mean, stdev

import pytest


def test_origin_windows_lags_and_panel_cardinality(feature_sql_case):
    run, origin, day = feature_sql_case
    expected_mean = mean(number % 5 for number in range(day - 6, day + 1))
    expected_std = stdev(number % 5 for number in range(day - 6, day + 1))
    expected_zero = sum(number % 5 == 0 for number in range(day - 27, day + 1)) / 28
    run(f"""
        CREATE TEMP TABLE panel AS SELECT * FROM build_forecast_features('{origin}');
        DO $$ BEGIN
            IF (SELECT COUNT(*) FROM panel) <> 56 THEN RAISE EXCEPTION 'Wrong panel size'; END IF;
            IF (SELECT COUNT(DISTINCT (unique_id, ds)) FROM panel) <> 56
                THEN RAISE EXCEPTION 'Duplicate panel keys'; END IF;
            IF EXISTS (SELECT 1 FROM panel WHERE ds <> forecast_origin + horizon)
                THEN RAISE EXCEPTION 'Wrong target date'; END IF;
            IF EXISTS (SELECT 1 FROM panel WHERE origin_lag_7 IS DISTINCT FROM ({((day - 7) % 5)}
                + CASE WHEN store_id = 'CA_2' THEN 100 ELSE 0 END))
                THEN RAISE EXCEPTION 'Lag 7 crossed origin or series'; END IF;
            IF EXISTS (SELECT 1 FROM panel WHERE origin_lag_14 IS DISTINCT FROM ({((day - 14) % 5)}
                + CASE WHEN store_id = 'CA_2' THEN 100 ELSE 0 END))
                THEN RAISE EXCEPTION 'Lag 14 wrong'; END IF;
            IF EXISTS (SELECT 1 FROM panel WHERE origin_lag_28 IS DISTINCT FROM ({((day - 28) % 5)}
                + CASE WHEN store_id = 'CA_2' THEN 100 ELSE 0 END))
                THEN RAISE EXCEPTION 'Lag 28 wrong'; END IF;
            IF EXISTS (SELECT 1 FROM panel WHERE rolling_mean_7 IS NULL
                OR ABS(rolling_mean_7 - {expected_mean}
                - CASE WHEN store_id = 'CA_2' THEN 100 ELSE 0 END) > 1e-10)
                THEN RAISE EXCEPTION 'Rolling mean boundary wrong'; END IF;
            IF EXISTS (SELECT 1 FROM panel WHERE rolling_std_7 IS NULL
                OR ABS(rolling_std_7 - {expected_std}) > 1e-10)
                THEN RAISE EXCEPTION 'Sample standard deviation wrong'; END IF;
            IF EXISTS (SELECT 1 FROM panel WHERE store_id = 'CA_1'
                AND (zero_sales_share_28 IS NULL
                    OR ABS(zero_sales_share_28 - {expected_zero}) > 1e-10))
                THEN RAISE EXCEPTION 'Zero share wrong'; END IF;
            IF EXISTS (SELECT 1 FROM panel WHERE available_days_7 <> 7 OR available_days_28 <> 28)
                THEN RAISE EXCEPTION 'Available-day counts wrong'; END IF;
        END $$;
    """)


def test_future_demand_and_prices_cannot_change_default_features(feature_sql_case):
    run, origin, _ = feature_sql_case
    run(f"""
        CREATE TEMP TABLE before_panel AS SELECT * FROM build_forecast_features('{origin}');
        UPDATE fact_daily_sales SET sales = sales + 10000 WHERE date > '{origin}';
        UPDATE dim_prices SET sell_price = sell_price + 10000 WHERE wm_yr_wk > 1055;
        CREATE TEMP TABLE after_panel AS SELECT * FROM build_forecast_features('{origin}');
        DO $$ BEGIN
            IF EXISTS ((SELECT * FROM before_panel EXCEPT ALL SELECT * FROM after_panel)
                UNION ALL (SELECT * FROM after_panel EXCEPT ALL SELECT * FROM before_panel))
                THEN RAISE EXCEPTION 'Post-origin data leaked into prediction inputs'; END IF;
        END $$;
    """)


def test_weekly_price_reference_and_explicit_known_price_scenario(feature_sql_case):
    run, origin, _ = feature_sql_case
    run(f"""
        CREATE TEMP TABLE panel AS SELECT * FROM build_forecast_features('{origin}');
        CREATE TEMP TABLE known AS SELECT * FROM build_forecast_features('{origin}', 28, TRUE);
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM panel WHERE sell_price IS DISTINCT FROM 65::FLOAT8
                OR price_mean_52w IS DISTINCT FROM 39.5::FLOAT8
                OR price_observed_weeks_52w IS DISTINCT FROM 52::BIGINT
                OR price_relativity IS NULL OR price_momentum IS NULL
                OR ABS(price_relativity - 65.0/39.5) > 1e-10
                OR ABS(price_momentum - (65.0/64 - 1)) > 1e-10)
                THEN RAISE EXCEPTION 'Weekly price reference wrong'; END IF;
            IF EXISTS (SELECT 1 FROM known WHERE horizon = 1
                AND sell_price IS DISTINCT FROM 66::FLOAT8)
                THEN RAISE EXCEPTION 'Known future price not used'; END IF;
            IF EXISTS (SELECT 1 FROM known WHERE NOT known_future_prices)
                THEN RAISE EXCEPTION 'Scenario not labeled'; END IF;
        END $$;
    """)


def test_missing_price_week_keeps_calendar_window_and_carries_only_past_price(feature_sql_case):
    run, origin, _ = feature_sql_case
    run(f"""
        DELETE FROM dim_prices WHERE store_id = 'CA_1' AND wm_yr_wk = 1055;
        CREATE TEMP TABLE panel AS SELECT * FROM build_forecast_features('{origin}');
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM panel WHERE store_id = 'CA_1' AND (
                sell_price IS DISTINCT FROM 64::FLOAT8
                OR price_mean_52w IS DISTINCT FROM 39::FLOAT8
                OR price_observed_weeks_52w <> 51 OR price_momentum IS NOT NULL
                OR NOT price_carried_forward))
                THEN RAISE EXCEPTION 'Missing week compressed history or backfilled prices'; END IF;
        END $$;
    """)


def test_no_price_history_does_not_backfill_from_future(feature_sql_case):
    run, origin, _ = feature_sql_case
    run(f"""
        DELETE FROM dim_prices WHERE wm_yr_wk <= 1055;
        CREATE TEMP TABLE panel AS SELECT * FROM build_forecast_features('{origin}');
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM panel WHERE sell_price IS NOT NULL
                OR price_relativity IS NOT NULL
                OR price_observed_weeks_52w IS DISTINCT FROM 0::BIGINT)
                THEN RAISE EXCEPTION 'Future price backfilled missing history'; END IF;
        END $$;
    """)


def test_price_momentum_does_not_skip_a_missing_previous_week(feature_sql_case):
    run, origin, _ = feature_sql_case
    run(f"""
        DELETE FROM dim_prices WHERE store_id = 'CA_1' AND wm_yr_wk = 1054;
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM build_forecast_features('{origin}', 1)
                WHERE store_id = 'CA_1' AND price_momentum IS NOT NULL)
                THEN RAISE EXCEPTION 'Momentum skipped an unlisted calendar week'; END IF;
        END $$;
    """)


def test_bounded_origin_snapshot_matches_historical_window_view(feature_sql_case):
    run, origin, _ = feature_sql_case
    run(f"""
        DO $$ BEGIN
            IF EXISTS (
                SELECT 1 FROM build_forecast_features('{origin}', 1) f
                JOIN feature_daily_demand d ON d.store_id = f.store_id AND d.item_id = f.item_id
                    AND d.date = f.forecast_origin
                WHERE ROW(f.origin_lag_7, f.origin_lag_14, f.origin_lag_28,
                    f.available_days_7, f.available_days_14, f.available_days_28)
                    IS DISTINCT FROM ROW(d.origin_lag_7, d.origin_lag_14, d.origin_lag_28,
                    d.available_days_7, d.available_days_14, d.available_days_28)
                OR ARRAY[f.rolling_mean_7, f.rolling_mean_14, f.rolling_mean_28,
                    f.rolling_std_7, f.rolling_std_14, f.rolling_std_28, f.zero_sales_share_28]
                    IS DISTINCT FROM ARRAY[d.rolling_mean_7::FLOAT8, d.rolling_mean_14::FLOAT8,
                    d.rolling_mean_28::FLOAT8, d.rolling_std_7::FLOAT8, d.rolling_std_14::FLOAT8,
                    d.rolling_std_28::FLOAT8, d.zero_sales_share_28::FLOAT8]
            ) THEN RAISE EXCEPTION 'Bounded aggregates drifted from historical windows'; END IF;
        END $$;
    """)


def test_national_holiday_countdown_calendar_and_snap(feature_sql_case):
    run, origin, day = feature_sql_case
    run(f"""
        CREATE TEMP TABLE panel AS SELECT * FROM build_forecast_features('{origin}');
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM panel WHERE horizon = 1
                AND days_until_next_holiday IS DISTINCT FROM 2)
                THEN RAISE EXCEPTION 'Countdown ignored second holiday slot'; END IF;
            IF EXISTS (SELECT 1 FROM panel WHERE horizon = 3 AND (
                days_until_next_holiday IS DISTINCT FROM 0
                OR NOT is_national_holiday OR NOT has_scheduled_event))
                THEN RAISE EXCEPTION 'Holiday day should have zero countdown'; END IF;
            IF EXISTS (SELECT 1 FROM panel WHERE horizon = 10 AND (
                is_national_holiday OR NOT has_scheduled_event
                OR days_until_next_holiday IS DISTINCT FROM 4))
                THEN RAISE EXCEPTION 'Non-national event counted as national holiday'; END IF;
            IF EXISTS (SELECT 1 FROM panel WHERE horizon > 14
                AND days_until_next_holiday IS NOT NULL)
                THEN RAISE EXCEPTION 'Unknown calendar tail was fabricated'; END IF;
            IF EXISTS (SELECT 1 FROM panel WHERE snap_ca <> (({day} + horizon) % 4 = 0)
                OR day_of_week <> EXTRACT(ISODOW FROM ds))
                THEN RAISE EXCEPTION 'Target calendar fields wrong'; END IF;
        END $$;
    """)


def test_unlisted_days_are_excluded_with_explicit_sample_counts(feature_sql_case):
    run, origin, day = feature_sql_case
    expected_zero = sum(number % 5 == 0 for number in range(day - 27, day - 6)) / 21
    run(f"""
        UPDATE fact_daily_sales SET sales = 0, is_available = FALSE
            WHERE store_id = 'CA_1' AND date BETWEEN DATE '{origin}' - 6 AND DATE '{origin}';
        CREATE TEMP TABLE panel AS SELECT * FROM build_forecast_features('{origin}');
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM panel WHERE store_id = 'CA_1' AND (
                rolling_mean_7 IS NOT NULL OR rolling_std_7 IS NOT NULL
                OR available_days_7 <> 0 OR available_days_28 <> 21 OR origin_is_available
                OR ABS(zero_sales_share_28 - {expected_zero}) > 1e-10))
                THEN RAISE EXCEPTION 'Unlisted days treated as observed zero demand'; END IF;
        END $$;
    """)


def test_missing_day_is_rejected_instead_of_compressing_lags(feature_sql_case):
    run, origin, _ = feature_sql_case
    run(
        f"""
        DELETE FROM fact_daily_sales WHERE store_id = 'CA_1' AND date = DATE '{origin}' - 10;
        SELECT * FROM build_forecast_features('{origin}');
    """,
        expected_error="29 complete observed calendar days",
    )


@pytest.mark.parametrize("horizon", [0, 29])
def test_invalid_horizon_is_rejected(feature_sql_case, horizon):
    run, origin, _ = feature_sql_case
    run(
        f"SELECT * FROM build_forecast_features('{origin}', {horizon});",
        expected_error="horizon between 1 and 28",
    )


def test_calendar_must_cover_entire_requested_horizon(feature_sql_case):
    run, origin, _ = feature_sql_case
    run(
        f"SELECT * FROM build_forecast_features('{origin + timedelta(days=29)}');",
        expected_error="Calendar does not cover",
    )


def test_feature_installation_is_repeatable_and_early_windows_remain_null(feature_sql_case):
    run, _, _ = feature_sql_case
    run(
        Path("sql/02_feature_view.sql").read_text()
        + """
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM feature_daily_demand WHERE date = (
                SELECT MIN(date) FROM fact_daily_sales
            ) AND (rolling_mean_7 IS NOT NULL OR rolling_std_28 IS NOT NULL))
                THEN RAISE EXCEPTION 'Insufficient warmup was filled'; END IF;
            IF (SELECT COUNT(*) FROM model_forecast_features) <> 56
                THEN RAISE EXCEPTION 'Default latest-origin view wrong'; END IF;
        END $$;
    """
    )
