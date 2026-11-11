"""Origin-safe feature snapshots for direct global forecasting.

PostgreSQL supplies the production feature views. The CSV backend reproduces
their definitions for immutable ingestion extracts and is checked against SQL.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from supply_chain_opt_engine.evaluation import validate_history
from supply_chain_opt_engine.features import install_feature_views, validate_panel
from supply_chain_opt_engine.history import ingestion_manifest, load_history
from supply_chain_opt_engine.m5 import checksum

DEMAND_COLUMNS = [
    "origin_lag_7",
    "origin_lag_14",
    "origin_lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
    "zero_sales_share_28",
    "available_days_7",
    "available_days_14",
    "available_days_28",
]
CATEGORICAL_COLUMNS = [
    "series_id",
    "store_id",
    "item_id",
    "dept_id",
    "cat_id",
    "event_name_1",
    "event_name_2",
]
NUMERIC_COLUMNS = [
    "horizon",
    *DEMAND_COLUMNS,
    "origin_is_available",
    "sell_price",
    "price_mean_52w",
    "price_relativity",
    "price_momentum",
    "price_observed_weeks_52w",
    "price_age_days",
    "price_carried_forward",
    "snap_ca",
    "is_national_holiday",
    "has_scheduled_event",
    "day_of_week",
    "month",
    "year",
    "day_of_month",
    "week_of_year",
    "days_until_next_holiday",
]


def calendar_features(calendar: pd.DataFrame) -> pd.DataFrame:
    frame = calendar.copy().sort_values("date")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["is_national_holiday"] = frame["event_type_1"].eq("National") | frame["event_type_2"].eq(
        "National"
    )
    frame["has_scheduled_event"] = frame[["event_name_1", "event_name_2"]].notna().any(axis=1)
    next_holiday = frame["date"].where(frame["is_national_holiday"]).bfill()
    frame["days_until_next_holiday"] = (next_holiday - frame["date"]).dt.days
    return frame


def demand_features(history: pd.DataFrame) -> pd.DataFrame:
    frame = history[["unique_id", "store_id", "item_id", "ds", "y", "is_available"]].copy()
    frame = frame.rename(columns={"ds": "date", "y": "sales"}).sort_values(["unique_id", "date"])
    frame = frame.reset_index(drop=True)
    frame["listed"] = frame["sales"].where(frame["is_available"])
    grouped = frame.groupby("unique_id", sort=False)
    age = grouped.cumcount() + 1
    for lag in [7, 14, 28]:
        frame[f"origin_lag_{lag}"] = grouped["listed"].shift(lag)
    for window in [7, 14, 28]:
        rolling = grouped["listed"].rolling(window, min_periods=1)
        frame[f"rolling_mean_{window}"] = (
            rolling.mean().reset_index(level=0, drop=True).where(age >= window)
        )
        frame[f"rolling_std_{window}"] = (
            rolling.std(ddof=1).reset_index(level=0, drop=True).where(age >= window)
        )
        frame[f"available_days_{window}"] = (
            rolling.count().reset_index(level=0, drop=True).fillna(0)
        )
    frame["zero"] = frame["sales"].eq(0).astype(float).where(frame["is_available"])
    frame["zero_sales_share_28"] = (
        frame.groupby("unique_id")["zero"]
        .rolling(28, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .where(age >= 28)
    )
    frame["observed_days_29"] = np.minimum(age, 29)
    return frame[
        [
            "unique_id",
            "store_id",
            "item_id",
            "date",
            "is_available",
            "observed_days_29",
            *DEMAND_COLUMNS,
        ]
    ]


def weekly_price_features(
    history: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    weeks = (
        calendar.groupby("wm_yr_wk", as_index=False)["date"]
        .min()
        .rename(columns={"date": "week_start"})
    )
    series = history[["unique_id", "store_id", "item_id"]].drop_duplicates()
    frame = (
        series.merge(weeks, how="cross")
        .merge(
            prices,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
            validate="one_to_one",
        )
        .sort_values(["unique_id", "week_start"])
        .reset_index(drop=True)
    )
    grouped = frame.groupby("unique_id", sort=False)
    rolling = grouped["sell_price"].rolling(52, min_periods=1)
    frame["price_mean_52w"] = rolling.mean().reset_index(level=0, drop=True)
    frame["price_observed_weeks_52w"] = rolling.count().reset_index(level=0, drop=True).fillna(0)
    previous = grouped["sell_price"].shift(1)
    frame["asof_price"] = grouped["sell_price"].ffill()
    frame["price_relativity"] = frame["asof_price"] / frame["price_mean_52w"]
    frame["price_momentum"] = frame["sell_price"] / previous - 1
    frame["observed_week"] = frame["week_start"].where(frame["sell_price"].notna())
    frame["last_price_week_start"] = frame.groupby("unique_id")["observed_week"].ffill()
    frame["price_carried_forward"] = frame["sell_price"].isna() & frame["asof_price"].notna()
    return frame.drop(columns=["sell_price", "observed_week"])


@dataclass
class FeatureStore:
    history: pd.DataFrame
    daily: pd.DataFrame
    calendar: pd.DataFrame
    weekly: pd.DataFrame
    items: pd.DataFrame
    provenance: dict

    def __post_init__(self):
        fingerprints = {}
        for name, frame, order in [
            ("history", self.history, ["unique_id", "ds"]),
            ("daily", self.daily, ["unique_id", "date"]),
            ("calendar", self.calendar, ["date"]),
            ("weekly", self.weekly, ["unique_id", "wm_yr_wk"]),
            ("items", self.items, ["item_id"]),
        ]:
            values = (
                pd.util.hash_pandas_object(frame.sort_values(order), index=False)
                .to_numpy()
                .tobytes()
            )
            fingerprints[name] = hashlib.sha256(values).hexdigest()
        self.provenance = {**self.provenance, "feature_snapshot_sha256": fingerprints}

    @classmethod
    def from_frames(cls, history, calendar, prices, items, provenance=None):
        history = validate_history(history)
        calendar = calendar_features(calendar)
        return cls(
            history,
            demand_features(history),
            calendar,
            weekly_price_features(history, calendar, prices),
            items[["item_id", "dept_id", "cat_id"]].drop_duplicates(),
            provenance or {},
        )

    def panel(
        self, origin: date, horizon: int = 28, known_future_prices: bool = False
    ) -> pd.DataFrame:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 28:
            raise ValueError("Feature horizon must lie between 1 and 28.")
        if not isinstance(known_future_prices, bool):
            raise ValueError("known_future_prices must be boolean.")
        origin_date = pd.Timestamp(origin)
        snapshot = self.daily[self.daily["date"].eq(origin_date)].copy()
        if (
            len(snapshot) != self.history["unique_id"].nunique()
            or snapshot["observed_days_29"].ne(29).any()
        ):
            raise ValueError("Every series needs 29 complete observed days through the origin.")
        target = self.calendar[
            self.calendar["date"].between(
                pd.Timestamp(origin + timedelta(days=1)),
                pd.Timestamp(origin + timedelta(days=horizon)),
            )
        ].copy()
        if len(target) != horizon:
            raise ValueError("Calendar does not cover the requested forecast horizon.")
        current_week = self.calendar.loc[self.calendar["date"].eq(origin_date), "wm_yr_wk"].item()
        snapshot = (
            snapshot.drop(columns=["observed_days_29"])
            .rename(columns={"date": "forecast_origin", "is_available": "origin_is_available"})
            .merge(self.items, on="item_id", validate="many_to_one")
        )
        target = target.rename(columns={"date": "ds"})
        frame = snapshot.merge(target, how="cross")
        frame["price_week"] = frame["wm_yr_wk"] if known_future_prices else current_week
        prices = self.weekly.drop(columns=["unique_id"]).rename(columns={"wm_yr_wk": "price_week"})
        frame = frame.merge(
            prices, on=["store_id", "item_id", "price_week"], how="left", validate="many_to_one"
        )
        frame["sell_price"] = frame["asof_price"]
        frame["price_age_days"] = (
            frame["ds"] - pd.to_datetime(frame["last_price_week_start"])
        ).dt.days
        frame["known_future_prices"] = known_future_prices
        frame["horizon"] = (frame["ds"] - origin_date).dt.days
        frame["day_of_week"] = frame["ds"].dt.dayofweek + 1
        frame["day_of_month"] = frame["ds"].dt.day
        frame["week_of_year"] = frame["ds"].dt.isocalendar().week.astype(int)
        columns = [
            "unique_id",
            "store_id",
            "item_id",
            "dept_id",
            "cat_id",
            "forecast_origin",
            "ds",
            *NUMERIC_COLUMNS,
            "known_future_prices",
            "event_name_1",
            "event_name_2",
        ]
        frame = (
            frame[list(dict.fromkeys(columns))]
            .sort_values(["unique_id", "ds"])
            .reset_index(drop=True)
        )
        validate_panel(frame, origin, horizon)
        return frame


def load_feature_store(
    root: Path, engine: Engine | None = None, manifest_path: Path | None = None
) -> FeatureStore:
    path, manifest = ingestion_manifest(root, manifest_path)
    if engine is None:
        history, provenance = load_history(root, manifest_path=path)
        frames = {}
        for name in ["dim_calendar", "dim_prices", "dim_sku_economics"]:
            filename = f"{name}.csv"
            if checksum(path.parent / filename) != manifest["artifact_sha256"][filename]:
                raise ValueError(f"Ingestion artifact checksum mismatch: {filename}")
            frames[name] = pd.read_csv(path.parent / filename)
        return FeatureStore.from_frames(
            history,
            frames["dim_calendar"],
            frames["dim_prices"],
            frames["dim_sku_economics"],
            provenance,
        )
    install_feature_views(engine, root / "sql/02_feature_view.sql")
    # Copy a single database snapshot before tuning; no query observes changing
    # source data halfway through a study or held-out evaluation.
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
        with connection.begin():
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            sales = pd.read_sql_query(text("SELECT * FROM fact_daily_sales"), connection)
            daily = pd.read_sql_query(text("SELECT * FROM feature_daily_demand"), connection)
            calendar = pd.read_sql_query(text("SELECT * FROM feature_calendar"), connection)
            weekly = pd.read_sql_query(text("SELECT * FROM feature_weekly_prices"), connection)
            items = pd.read_sql_query(
                text("SELECT item_id, dept_id, cat_id FROM dim_sku_economics"), connection
            )
    for frame in [sales, daily, weekly]:
        frame["unique_id"] = frame["store_id"] + "/" + frame["item_id"]
    history = validate_history(sales.rename(columns={"date": "ds", "sales": "y"}))
    selected = pd.read_csv(path.parent / "selected_series.csv")
    if (
        set(history["unique_id"]) != set(selected["unique_id"])
        or len(history) != manifest["sales_rows"]
    ):
        raise ValueError("PostgreSQL history does not match the committed scope.")
    for frame in [daily, calendar]:
        frame["date"] = pd.to_datetime(frame["date"])
    provenance = {
        "source_type": "postgresql_feature_snapshot",
        "ingestion_run_id": manifest["run_id"],
        "manifest_sha256": checksum(path),
        "selection_cutoff": manifest["config"]["selection_cutoff"],
    }
    return FeatureStore(history, daily, calendar, weekly, items, provenance)


@dataclass
class FeatureEncoder:
    categories: dict[str, list[str]]

    @classmethod
    def from_store(cls, store: FeatureStore):
        ids = store.history["unique_id"].unique()
        series = (
            store.history[["store_id", "item_id"]]
            .drop_duplicates()
            .merge(store.items, on="item_id")
        )
        categories = {"series_id": sorted(ids.tolist())}
        for name in ["store_id", "item_id", "dept_id", "cat_id"]:
            categories[name] = sorted(series[name].unique().tolist())
        for name in ["event_name_1", "event_name_2"]:
            categories[name] = sorted(store.calendar[name].dropna().astype(str).unique().tolist())
        return cls(categories)

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(index=panel.index)
        for name in CATEGORICAL_COLUMNS:
            values = panel["unique_id"] if name == "series_id" else panel[name]
            frame[name] = pd.Categorical(values, categories=self.categories[name])
        for name in NUMERIC_COLUMNS:
            frame[name] = pd.to_numeric(panel[name]).astype("float32")
        if np.isinf(frame[NUMERIC_COLUMNS].to_numpy()).any():
            raise ValueError("Infinite model feature value.")
        return frame.reset_index(drop=True)
