"""The immutable-CSV feature backend must match the production SQL contract."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from supply_chain_opt_engine.feature_store import FeatureStore


@pytest.mark.parametrize("scenario", ["regular", "missing_week", "unlisted"])
@pytest.mark.parametrize("known_prices", [False, True])
def test_csv_feature_store_matches_postgresql(feature_sql_case, tmp_path, scenario, known_prices):
    run, origin, _ = feature_sql_case
    calendar = pd.read_csv(tmp_path / "dim_calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(tmp_path / "dim_prices.csv")
    sales = pd.read_csv(tmp_path / "fact_daily_sales.csv", parse_dates=["date"])
    items = pd.read_csv(tmp_path / "dim_sku_economics.csv")
    if scenario != "regular":
        prices = prices.loc[~(prices.store_id.eq("CA_1") & prices.wm_yr_wk.eq(1055))]
        prices.to_csv(tmp_path / "dim_prices.csv", index=False)
    if scenario == "unlisted":
        mask = sales.store_id.eq("CA_1") & sales.date.between(
            pd.Timestamp(origin - timedelta(days=6)), pd.Timestamp(origin)
        )
        sales.loc[mask, "sales"] = 0
        sales.loc[mask, "is_available"] = False
        sales.to_csv(tmp_path / "fact_daily_sales.csv", index=False)
    sales["unique_id"] = sales["store_id"] + "/" + sales["item_id"]
    store = FeatureStore.from_frames(
        sales.rename(columns={"date": "ds", "sales": "y"}), calendar, prices, items
    )
    expected_path = tmp_path / "expected_features.csv"
    run(
        f"COPY (SELECT * FROM build_forecast_features('{origin}', 28, {str(known_prices).upper()}) "
        f"ORDER BY unique_id, ds) TO '{expected_path}' WITH (FORMAT CSV, HEADER TRUE);"
    )
    expected = pd.read_csv(
        expected_path, parse_dates=["forecast_origin", "ds"], true_values=["t"], false_values=["f"]
    )
    actual = store.panel(origin, 28, known_future_prices=known_prices)
    assert_frame_equal(actual[expected.columns], expected, check_dtype=False, atol=1e-8, rtol=1e-8)
