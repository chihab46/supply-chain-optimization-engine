"""Validate M5 source files and select a reproducible store-item panel."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_FILES = ("sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv")
IDENTIFIERS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_COLUMNS = [
    "date",
    "d",
    "wm_yr_wk",
    "weekday",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
]


@dataclass(frozen=True)
class ScopeConfig:
    """Selection is based solely on sales through selection_cutoff, inclusive."""

    selection_cutoff: date
    stores: tuple[str, ...] = ("CA_1", "CA_2", "CA_3")
    categories: tuple[str, ...] = ("FOODS", "HOUSEHOLD", "HOBBIES")
    state: str = "CA"
    max_bottom_series: int = 600
    history_days: int = 1100
    seed: int = 42
    chunk_size: int = 512

    def __post_init__(self) -> None:
        for name in ["max_bottom_series", "history_days", "seed", "chunk_size"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer.")
        if self.state != "CA" or not set(self.stores) <= {"CA_1", "CA_2", "CA_3"}:
            raise ValueError("This schema supports CA_1, CA_2 and CA_3 only.")
        if not set(self.categories) <= {"FOODS", "HOUSEHOLD", "HOBBIES"}:
            raise ValueError("Unsupported M5 category.")
        for values in (self.stores, self.categories):
            if not values or len(values) != len(set(values)):
                raise ValueError("Stores and categories must be nonempty and unique.")
        if self.max_bottom_series < len(self.stores) * len(self.categories):
            raise ValueError("The series limit must cover every store-category combination.")
        if self.history_days < 29 or self.chunk_size < 1 or self.seed < 0:
            raise ValueError("Require history_days >= 29, chunk_size > 0 and seed >= 0.")

    def manifest(self) -> dict:
        return {**asdict(self), "selection_cutoff": self.selection_cutoff.isoformat()}


@dataclass
class ScopedDataset:
    calendar: pd.DataFrame
    selected: pd.DataFrame
    sales: pd.DataFrame
    prices: pd.DataFrame
    config: ScopeConfig
    source_checksums: dict[str, str]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")


def require_keys(frame: pd.DataFrame, keys: list[str], source: str) -> None:
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{source}: null or duplicate key {keys}")


def integers(values: pd.DataFrame | pd.Series, source: str, minimum: int = 0):
    array = values.to_numpy(dtype="float64")
    if (
        not np.isfinite(array).all()
        or (array < minimum).any()
        or (array > np.iinfo(np.int32).max).any()
        or (array != np.floor(array)).any()
    ):
        raise ValueError(f"{source}: expected finite integers >= {minimum} within int32 range")
    return values.astype("int32")


def read_calendar(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, CALENDAR_COLUMNS, "calendar")
    frame = frame[CALENDAR_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    day_numbers = frame["d"].astype(str).str.extract(r"^d_([1-9][0-9]*)$", expand=False)
    frame["d"] = integers(pd.to_numeric(day_numbers, errors="raise"), "calendar day", 1)
    for column in ["wm_yr_wk", "wday", "month", "year", "snap_CA"]:
        frame[column] = integers(frame[column], f"calendar {column}")
    require_keys(frame, ["date"], "calendar")
    require_keys(frame, ["d"], "calendar")
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty or not frame["date"].diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Calendar must be a nonempty, complete daily grid.")
    if not frame["d"].diff().dropna().eq(1).all():
        raise ValueError("Calendar day IDs must increase one per date.")
    if (
        not frame["month"].eq(frame["date"].dt.month).all()
        or not frame["year"].eq(frame["date"].dt.year).all()
        or not frame["wday"].between(1, 7).all()
        or not frame["snap_CA"].isin([0, 1]).all()
        or frame["weekday"].isna().any()
    ):
        raise ValueError("Calendar fields disagree with dates or have invalid values.")
    return frame.rename(columns={"snap_CA": "snap_ca"}).assign(
        snap_ca=frame["snap_CA"].astype(bool)
    )


def read_sales(
    path: Path, calendar: pd.DataFrame, config: ScopeConfig
) -> tuple[pd.DataFrame, list[str]]:
    # Reject raw duplicate headers before pandas can silently rename them.
    with path.open(newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError("sales: duplicate column names")
    require_columns(pd.DataFrame(columns=header), IDENTIFIERS, "sales")
    all_days = [column for column in header if column.startswith("d_")]
    try:
        all_days.sort(key=lambda value: int(value[2:]))
    except ValueError as error:
        raise ValueError("sales: invalid M5 day column") from error
    if len(all_days) < config.history_days:
        raise ValueError("Sales history is shorter than history_days.")
    day_columns = all_days[-config.history_days :]
    numbers = [int(column[2:]) for column in day_columns]
    if numbers != list(range(numbers[0], numbers[-1] + 1)):
        raise ValueError("Sales day columns must form a complete daily grid.")
    lookup = calendar.set_index("d")["date"]
    if not set(numbers) <= set(lookup.index):
        raise ValueError("Sales day columns are missing from calendar.")
    cutoff = pd.Timestamp(config.selection_cutoff)
    if not lookup[numbers[27]] <= cutoff < lookup[numbers[-1]]:
        raise ValueError("Selection cutoff needs >= 28 training days and later evaluation data.")
    chunks = []
    for chunk in pd.read_csv(path, usecols=IDENTIFIERS + day_columns, chunksize=config.chunk_size):
        chunk = chunk.loc[
            chunk["store_id"].isin(config.stores) & chunk["cat_id"].isin(config.categories)
        ].copy()
        if chunk.empty:
            continue
        if not chunk["state_id"].eq(config.state).all() or chunk[IDENTIFIERS].isna().any().any():
            raise ValueError("sales: missing identifiers or store/state mismatch")
        chunk[day_columns] = integers(chunk[day_columns], "sales")
        chunks.append(chunk)
    if not chunks:
        raise ValueError("No sales series match the requested scope.")
    frame = pd.concat(chunks, ignore_index=True)
    require_keys(frame, ["store_id", "item_id"], "sales")
    for column in ["cat_id", "dept_id"]:
        if frame.groupby("item_id")[column].nunique().gt(1).any():
            raise ValueError(f"sales: conflicting {column} for the same item")
    if not frame.apply(lambda row: row["dept_id"].startswith(row["cat_id"] + "_"), axis=1).all():
        raise ValueError("sales: department/category mismatch")
    return frame, day_columns


def read_prices(path: Path, candidates: pd.DataFrame, config: ScopeConfig) -> pd.DataFrame:
    keys = ["store_id", "item_id"]
    chunks = []
    for chunk in pd.read_csv(path, chunksize=max(config.chunk_size, 10000)):
        require_columns(chunk, [*keys, "wm_yr_wk", "sell_price"], "prices")
        chunk = chunk.merge(candidates[keys], on=keys, how="inner", validate="many_to_one")
        if not chunk.empty:
            chunks.append(chunk[[*keys, "wm_yr_wk", "sell_price"]])
    if not chunks:
        raise ValueError("No prices match the scoped series.")
    frame = pd.concat(chunks, ignore_index=True)
    frame["wm_yr_wk"] = integers(frame["wm_yr_wk"], "price week", 1)
    price = pd.to_numeric(frame["sell_price"], errors="raise")
    if not np.isfinite(price).all() or not price.between(0.0001, 99999999.9999).all():
        raise ValueError("Prices must be finite and positive within NUMERIC(12,4) range.")
    frame["sell_price"] = price.round(4)
    require_keys(frame, [*keys, "wm_yr_wk"], "prices")
    return frame


def select_series(
    wide: pd.DataFrame,
    days: list[str],
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    config: ScopeConfig,
) -> pd.DataFrame:
    """Round-robin sampling across stores, categories and observed demand types.

    Use only listed weeks through the cutoff, so unlisted periods do not inflate
    intermittency. Take the fastest series in each stratum first, then interleave
    ranked candidates until the panel limit is reached.
    """
    dates = calendar.set_index("d")
    train = [
        day for day in days if dates.loc[int(day[2:]), "date"].date() <= config.selection_cutoff
    ]
    weeks = dates.loc[[int(day[2:]) for day in train], "wm_yr_wk"].to_numpy()
    price_keys = set(prices[["store_id", "item_id", "wm_yr_wk"]].itertuples(index=False, name=None))
    stats = []
    for row in wide.itertuples(index=False):
        # Identifier order follows the DataFrame columns, not source CSV order.
        metadata = dict(zip(wide.columns, row, strict=True))
        listed = np.array(
            [(metadata["store_id"], metadata["item_id"], int(w)) in price_keys for w in weeks]
        )
        observed = np.array([metadata[day] for day in train])[listed]
        if len(observed) == 0 or observed.sum() == 0:
            continue  # No eligible pre-cutoff demand, including products introduced later.
        zero_share = float((observed == 0).mean())
        stats.append(
            {
                **{column: metadata[column] for column in IDENTIFIERS},
                "mean_daily_sales": float(observed.mean()),
                "zero_sales_share": zero_share,
                "training_available_days": int(len(observed)),
                "demand_type": "intermittent" if zero_share >= 0.5 else "fast",
            }
        )
    if not stats:
        raise ValueError("No series have listed, positive demand before the selection cutoff.")
    ranked = pd.DataFrame(stats)
    expected = {(store, category) for store in config.stores for category in config.categories}
    actual = set(ranked[["store_id", "cat_id"]].itertuples(index=False, name=None))
    if expected - actual:
        raise ValueError(
            f"No eligible training series in store/category groups: {sorted(expected - actual)}"
        )
    # Stable hash breaks velocity ties without depending on source row order.
    ranked["tie"] = ranked.apply(
        lambda row: hashlib.sha256(
            f"{config.seed}:{row.store_id}:{row.item_id}".encode()
        ).hexdigest(),
        axis=1,
    )
    ranked = ranked.sort_values(["mean_daily_sales", "tie"], ascending=[False, True])
    groups = ["store_id", "cat_id", "demand_type"]
    ranked["rank"] = ranked.groupby(groups).cumcount()
    # First cover all store-category cells, then all available demand strata.
    leaders = ranked.drop_duplicates(["store_id", "cat_id"])
    remaining = ranked.loc[~ranked.index.isin(leaders.index)].sort_values(["rank", *groups, "tie"])
    selected = pd.concat([leaders, remaining]).head(config.max_bottom_series)
    selected = selected.drop(columns=["rank", "tie"]).sort_values(["store_id", "item_id"])
    selected["unique_id"] = selected["store_id"] + "/" + selected["item_id"]
    return selected.reset_index(drop=True)


def prepare_dataset(raw_dir: Path, config: ScopeConfig) -> ScopedDataset:
    missing = [name for name in SOURCE_FILES if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing M5 files in {raw_dir}: {missing}. "
            "Run the download subcommand after configuring Kaggle, or copy the CSVs there."
        )
    checksums = {name: checksum(raw_dir / name) for name in SOURCE_FILES}
    calendar = read_calendar(raw_dir / "calendar.csv")
    wide, days = read_sales(raw_dir / "sales_train_evaluation.csv", calendar, config)
    prices = read_prices(raw_dir / "sell_prices.csv", wide, config)
    selected = select_series(wide, days, calendar, prices, config)
    keys = ["store_id", "item_id"]
    prices = prices.merge(selected[keys], on=keys, validate="many_to_one")
    selected_wide = wide.merge(selected[keys], on=keys, validate="one_to_one")
    sales = selected_wide.melt(id_vars=keys, value_vars=days, var_name="d", value_name="sales")
    sales["d"] = sales["d"].str[2:].astype("int32")
    sales = sales.merge(calendar[["d", "date", "wm_yr_wk"]], on="d", validate="many_to_one")
    sales = sales.merge(prices, on=[*keys, "wm_yr_wk"], how="left", validate="many_to_one")
    sales["is_available"] = sales["sell_price"].notna()
    if (sales["sales"].gt(0) & ~sales["is_available"]).any():
        raise ValueError("Positive sales without a listed weekly price in selected data.")
    sales = sales[[*keys, "date", "sales", "is_available"]].sort_values([*keys, "date"])
    require_keys(sales, [*keys, "date"], "scoped sales")
    if len(sales) != len(selected) * config.history_days:
        raise ValueError("Scoped sales are not a complete series-date grid.")
    # Preserve future known calendar and all scoped price weeks. Future prices
    # remain raw observations; feature construction must enforce availability.
    if checksums != {name: checksum(raw_dir / name) for name in SOURCE_FILES}:
        raise ValueError("Source files changed during ingestion; retry with immutable inputs.")
    return ScopedDataset(calendar, selected, sales, prices, config, checksums)
