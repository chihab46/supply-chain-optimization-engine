"""Reproducible synthetic SKU economics derived from training-period prices."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from supply_chain_opt_engine.m5 import ScopedDataset

ECONOMICS_PARAMETERS = {
    "currency": "USD",
    "purchase_price_fraction": [0.45, 0.75],
    "annual_holding_rate": [0.20, 0.35],
    "stockout_to_holding_ratio": [3.1, 7.9],
    "unit_volume_m3": {
        "FOODS": [0.0003, 0.003],
        "HOUSEHOLD": [0.001, 0.02],
        "HOBBIES": [0.0005, 0.015],
    },
    "lead_time_days": [2, 3, 5],
    "minimum_order_quantities": [6, 12, 24],
    "fixed_order_cost": [2.0, 12.0],
}


def generate_economics(dataset: ScopedDataset) -> pd.DataFrame:
    """Generate one row per item, shared across stores.

    Prices, costs, and fees are USD; volumes are m3. These ranges are modeling
    assumptions, not estimates of Walmart's actual costs.
    """
    cutoff = pd.Timestamp(dataset.config.selection_cutoff)
    first_dates = dataset.calendar.groupby("wm_yr_wk", as_index=False)["date"].min()
    known = dataset.prices.merge(first_dates, on="wm_yr_wk", validate="many_to_one")
    known = known.loc[known["date"] <= cutoff]
    reference_prices = known.groupby("item_id")["sell_price"].median()
    parameters = ECONOMICS_PARAMETERS
    rows = []
    items = (
        dataset.selected[["item_id", "dept_id", "cat_id"]].drop_duplicates().sort_values("item_id")
    )
    for item in items.itertuples(index=False):
        digest = hashlib.sha256(f"{dataset.config.seed}:{item.item_id}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8]))
        cost = max(
            round(
                float(reference_prices[item.item_id])
                * rng.uniform(*parameters["purchase_price_fraction"]),
                4,
            ),
            0.0001,
        )
        holding = max(
            round(cost * rng.uniform(*parameters["annual_holding_rate"]) / 365, 6), 0.000001
        )
        rows.append(
            {
                "item_id": item.item_id,
                "dept_id": item.dept_id,
                "cat_id": item.cat_id,
                "unit_purchase_cost": cost,
                "daily_holding_cost": holding,
                "stockout_penalty": round(
                    holding * rng.uniform(*parameters["stockout_to_holding_ratio"]), 6
                ),
                "unit_volume_m3": round(rng.uniform(*parameters["unit_volume_m3"][item.cat_id]), 6),
                "lead_time_days": int(rng.choice(parameters["lead_time_days"])),
                "minimum_order_quantity": int(rng.choice(parameters["minimum_order_quantities"])),
                "fixed_order_cost": round(rng.uniform(*parameters["fixed_order_cost"]), 4),
            }
        )
    return pd.DataFrame(rows)
