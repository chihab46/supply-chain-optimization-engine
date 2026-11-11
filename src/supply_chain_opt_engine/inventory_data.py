"""Validated planning arrays and reproducible pre-origin inventory scenarios."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from supply_chain_opt_engine.m5 import checksum

COST_COLUMNS = [
    "unit_purchase_cost",
    "daily_holding_cost",
    "stockout_penalty",
    "unit_volume_m3",
    "fixed_order_cost",
]


@dataclass(frozen=True)
class InventoryInputs:
    catalog: pd.DataFrame
    dates: pd.DatetimeIndex
    origin: pd.Timestamp
    decision_days: int
    demand: np.ndarray
    initial: np.ndarray
    receipts: np.ndarray
    capacity: dict[str, float]

    def __post_init__(self):
        c = self.catalog
        required = [
            "unique_id",
            "store_id",
            "item_id",
            "lead_time_days",
            "minimum_order_quantity",
            *COST_COLUMNS,
        ]
        if set(required) - set(c) or c.empty or c[required].isna().any().any():
            raise ValueError("Inventory catalog requires complete IDs and economics.")
        if c.unique_id.duplicated().any() or c.duplicated(["store_id", "item_id"]).any():
            raise ValueError("Inventory series and store-item keys must be unique.")
        for column in [*COST_COLUMNS, "lead_time_days", "minimum_order_quantity"]:
            values = c[column].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid nonnegative economics: {column}")
        if c.unit_volume_m3.le(0).any():
            raise ValueError("Item volumes must be positive.")
        for column in ["lead_time_days", "minimum_order_quantity"]:
            values = c[column].to_numpy(dtype=float)
            if (values < 1).any() or not np.equal(values, np.floor(values)).all():
                raise ValueError("Lead times and MOQs must be positive integers.")
        n, h = len(c), len(self.dates)
        if not isinstance(self.decision_days, int) or not 1 <= self.decision_days <= h:
            raise ValueError("Decision horizon must fit within the evaluation horizon.")
        expected = pd.date_range(pd.Timestamp(self.origin) + pd.Timedelta(days=1), periods=h)
        if h == 0 or not self.dates.equals(expected):
            raise ValueError("Planning dates must be a complete daily grid after the origin.")
        for name, shape in [("demand", (n, h)), ("receipts", (n, h)), ("initial", (n,))]:
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"Invalid finite nonnegative {name} with shape {shape}.")
            if name != "demand" and not np.equal(value, np.floor(value)).all():
                raise ValueError("Starting inventory and pipeline receipts must be integer units.")
        if set(self.capacity) != set(c.store_id):
            raise ValueError("Every store requires exactly one capacity.")
        if not np.isfinite(list(self.capacity.values())).all() or min(self.capacity.values()) < 0:
            raise ValueError("Store capacities must be finite and nonnegative.")

    @property
    def n(self):
        return len(self.catalog)

    @property
    def h(self):
        return len(self.dates)

    def with_demand(self, demand):
        return replace(self, demand=np.asarray(demand, dtype=float))

    def for_store(self, store_id):
        mask = self.catalog.store_id.eq(store_id).to_numpy()
        if not mask.any():
            raise ValueError("Unknown store.")
        return replace(
            self,
            catalog=self.catalog.loc[mask].reset_index(drop=True),
            demand=self.demand[mask],
            initial=self.initial[mask],
            receipts=self.receipts[mask],
            capacity={store_id: self.capacity[store_id]},
        )


def extend_weekly(demand: np.ndarray, tail_days: int):
    """Repeat the final seven forecast days; never substitute future actual demand."""
    if tail_days < 0 or demand.ndim != 2 or demand.shape[1] < 1:
        raise ValueError("Invalid forecast tail.")
    width = min(7, demand.shape[1])
    return np.concatenate([demand, demand[:, -width:][:, np.arange(tail_days) % width]], axis=1)


def forecast_matrix(frame, ids, dates, column):
    if frame.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Forecast keys must be unique before inventory planning.")
    if set(frame.unique_id) != set(ids) or set(pd.to_datetime(frame.ds)) != set(dates):
        raise ValueError("Forecast IDs or dates differ from the inventory scope.")
    values = (
        frame.pivot(index="unique_id", columns="ds", values=column)
        .reindex(index=ids, columns=dates)
        .to_numpy(dtype=float)
    )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Forecast demand must cover the full nonnegative panel.")
    return values


def generate_scenario(
    history,
    economics,
    nodes,
    origin,
    demand,
    *,
    tail_days=14,
    seed=42,
    capacity_multiplier=1.0,
    lead_time_shock=0,
):
    """Build scenario side inputs using the latest 56 observed days only."""
    origin = pd.Timestamp(origin)
    catalog = nodes[["unique_id", "store_id", "item_id"]].copy().sort_values("unique_id")
    catalog = catalog.merge(economics, on="item_id", validate="many_to_one").reset_index(drop=True)
    ids = catalog.unique_id.tolist()
    train = history[history.ds.between(origin - pd.Timedelta(days=55), origin)]
    recent = train.pivot(index="unique_id", columns="ds", values="y").reindex(ids)
    if recent.shape != (len(ids), 56) or recent.isna().any().any():
        raise ValueError("Inventory scenarios need 56 complete observed days through the origin.")
    mean = recent.mean(axis=1).to_numpy()
    catalog["recent_mean"] = mean
    catalog["recent_std"] = recent.std(axis=1, ddof=1).to_numpy()
    catalog["recent_p90"] = recent.quantile(0.9, axis=1).to_numpy()
    base_lead = catalog.lead_time_days.to_numpy(dtype=int)
    catalog["lead_time_days"] = base_lead + lead_time_shock
    main_h = demand.shape[1]
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=main_h + tail_days)
    initial = np.zeros(len(ids), dtype=int)
    receipts = np.zeros((len(ids), len(dates)), dtype=int)
    pipeline_rows = []
    for i, uid in enumerate(ids):
        token = hashlib.sha256(f"inventory-v1:{seed}:{origin.date()}:{uid}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(token[:8]))
        initial[i] = int(np.ceil(mean[i] * base_lead[i] * rng.uniform(0.75, 1.25)))
        units = int(np.ceil(mean[i] * base_lead[i] * rng.uniform(0.5, 1.0)))
        day = int(rng.integers(0, base_lead[i]))
        arrival = origin + pd.Timedelta(days=1 + day + lead_time_shock)
        if units:
            if day + lead_time_shock >= len(dates):
                raise ValueError("Planning horizon omits a known pipeline receipt.")
            receipts[i, day + lead_time_shock] = units
            pipeline_rows.append(
                {
                    "unique_id": uid,
                    "order_date": str((origin + pd.Timedelta(days=1 + day - base_lead[i])).date()),
                    "arrival_date": str(arrival.date()),
                    "units": units,
                }
            )
    # Fixed baseline capacities are independent of forecast quantile and lead-time shock.
    volume = catalog.unit_volume_m3.to_numpy()
    reference = np.maximum(mean, catalog.recent_p90.to_numpy()) * (base_lead + 7)
    reference += catalog.minimum_order_quantity.to_numpy()
    capacity = {}
    for store, index in catalog.groupby("store_id").groups.items():
        loc = np.asarray(index)
        base = (
            max(
                float(np.dot(reference[loc], volume[loc])), float(np.dot(initial[loc], volume[loc]))
            )
            * 1.25
        )
        capacity[store] = base * capacity_multiplier
    inputs = InventoryInputs(
        catalog,
        dates,
        origin,
        main_h,
        extend_weekly(demand, tail_days),
        initial,
        receipts,
        capacity,
    )
    # Hash only the training window used to generate inventory state, not future targets.
    train_hash = hashlib.sha256(
        pd.util.hash_pandas_object(recent, index=True).to_numpy().tobytes()
    ).hexdigest()
    metadata = {
        "protocol": "inventory-v1",
        "origin": str(origin.date()),
        "seed": seed,
        "training_days": 56,
        "training_start": str(recent.columns.min().date()),
        "training_end": str(origin.date()),
        "preorigin_demand_sha256": train_hash,
        "capacity_multiplier": capacity_multiplier,
        "lead_time_shock_days": lead_time_shock,
        "decision_days": main_h,
        "tail_days": tail_days,
        "initial_policy": "ceil(recent mean * base lead * seeded U[.75,1.25])",
        "pipeline_policy": "ceil(mean * base lead * seeded U[.5,1]); pre-origin orders",
        "capacity_policy": "1.25 * sum((max(mean,P90)*(base lead+7)+MOQ)*volume)",
        "tail_policy": "Repeat last forecast week; no new orders in tail; zero salvage credit",
    }
    pipeline = pd.DataFrame(
        pipeline_rows, columns=["unique_id", "order_date", "arrival_date", "units"]
    )
    return inputs, metadata, pipeline


def save_scenario(inputs, metadata, pipeline, directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    inputs.catalog.to_csv(directory / "catalog.csv", index=False)
    pd.DataFrame({"unique_id": inputs.catalog.unique_id, "units": inputs.initial}).to_csv(
        directory / "starting_inventory.csv", index=False
    )
    pipeline.to_csv(directory / "pipeline_receipts.csv", index=False)
    pd.DataFrame([{"store_id": s, "capacity_m3": c} for s, c in inputs.capacity.items()]).to_csv(
        directory / "store_capacity.csv", index=False
    )
    manifest = {**metadata, "artifact_sha256": {p.name: checksum(p) for p in directory.iterdir()}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
