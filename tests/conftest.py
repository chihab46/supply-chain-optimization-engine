"""Small M5-shaped sources with fast, intermittent and late-introduced items."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from supply_chain_opt_engine.m5 import ScopeConfig


@pytest.fixture
def history_panel():
    dates = pd.date_range("2019-01-01", periods=800)
    day = np.arange(len(dates))
    return pd.DataFrame(
        {
            "unique_id": ["A"] * len(dates) + ["B"] * len(dates),
            "ds": list(dates) * 2,
            "y": np.concatenate([1 + day % 7 + (day // 7) * 0.1, np.where(day % 7 == 0, 2, 0)]),
            "is_available": True,
        }
    )


@pytest.fixture
def m5_raw(tmp_path):
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    dates = pd.date_range("2020-01-01", periods=56)
    calendar = pd.DataFrame(
        {
            "date": dates,
            "d": [f"d_{day}" for day in range(1, 57)],
            "wm_yr_wk": [100 + day // 7 for day in range(56)],
            "weekday": dates.day_name(),
            "wday": dates.dayofweek + 1,
            "month": dates.month,
            "year": dates.year,
            "event_name_1": None,
            "event_type_1": None,
            "event_name_2": None,
            "event_type_2": None,
            "snap_CA": [int(day % 7 == 0) for day in range(56)],
        }
    )
    calendar.to_csv(raw / "calendar.csv", index=False)
    sales, prices = [], []
    for store in ["CA_1", "CA_2", "CA_3"]:
        for category in ["FOODS", "HOUSEHOLD", "HOBBIES"]:
            for item in range(5):
                item_id = f"{category}_1_{item:03}"
                row = {
                    "id": f"{item_id}_{store}_evaluation",
                    "item_id": item_id,
                    "dept_id": f"{category}_1",
                    "cat_id": category,
                    "store_id": store,
                    "state_id": "CA",
                }
                for day in range(1, 43):
                    value = (8 - item) if item < 2 else (2 if day % 7 == 0 else 0)
                    if item == 4:
                        value = 50 if day > 28 else 0
                    row[f"d_{day}"] = value
                sales.append(row)
                for week in range(100, 108):
                    if item == 4 and week < 104:
                        continue
                    prices.append(
                        {
                            "store_id": store,
                            "item_id": item_id,
                            "wm_yr_wk": week,
                            "sell_price": 3.0 + item,
                        }
                    )
    pd.DataFrame(sales).to_csv(raw / "sales_train_evaluation.csv", index=False)
    pd.DataFrame(prices).to_csv(raw / "sell_prices.csv", index=False)
    return raw


@pytest.fixture
def scope():
    return ScopeConfig(
        selection_cutoff=date(2020, 1, 28), max_bottom_series=18, history_days=42, chunk_size=7
    )


@pytest.fixture
def model_inputs(history_panel):
    history = history_panel.copy()
    history["store_id"] = history.unique_id.map({"A": "CA_1", "B": "CA_2"})
    history["item_id"] = "FOODS_1_001"
    dates = pd.date_range(history.ds.min(), periods=830)
    calendar = pd.DataFrame(
        {
            "date": dates,
            "d": np.arange(1, len(dates) + 1),
            "wm_yr_wk": 1000 + np.arange(len(dates)) // 7,
            "weekday": dates.day_name(),
            "wday": dates.dayofweek + 1,
            "month": dates.month,
            "year": dates.year,
            "event_name_1": [
                ("Holiday" if index % 90 == 0 else None) for index in range(len(dates))
            ],
            "event_type_1": [
                ("National" if index % 90 == 0 else None) for index in range(len(dates))
            ],
            "event_name_2": None,
            "event_type_2": None,
            "snap_ca": np.arange(len(dates)) % 5 == 0,
        }
    )
    prices = pd.DataFrame(
        [
            {
                "store_id": store,
                "item_id": "FOODS_1_001",
                "wm_yr_wk": week,
                "sell_price": 10 + (week % 5),
            }
            for store in ["CA_1", "CA_2"]
            for week in calendar.wm_yr_wk.unique()
        ]
    )
    items = pd.DataFrame({"item_id": ["FOODS_1_001"], "dept_id": ["FOODS_1"], "cat_id": ["FOODS"]})
    return history, calendar, prices, items
