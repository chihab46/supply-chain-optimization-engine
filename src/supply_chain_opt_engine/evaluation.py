"""Chronological fold definitions, sales validation and training-only scales."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

SEASONS = ("spring", "summer", "back_to_school", "winter_holidays")
SEASON_MONTHS = ({3, 4, 5}, {6, 7}, {8, 9}, {11, 12, 1})


@dataclass(frozen=True)
class Fold:
    fold_id: int
    origin: date
    horizon: int
    season: str
    role: str

    @property
    def validation_start(self) -> date:
        return self.origin + timedelta(days=1)

    @property
    def validation_end(self) -> date:
        return self.origin + timedelta(days=self.horizon)

    def metadata(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "origin": self.origin.isoformat(),
            "validation_start": self.validation_start.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "horizon": self.horizon,
            "season": self.season,
            "role": self.role,
        }


def seasonal_folds(cutoffs: list[date], horizon: int, selection_cutoff: date) -> list[Fold]:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Forecast horizon must be a positive integer.")
    if len(cutoffs) != 4 or cutoffs != sorted(set(cutoffs)):
        raise ValueError("Freeze four distinct, chronologically ordered cutoff dates.")
    if selection_cutoff > cutoffs[0]:
        raise ValueError("Series selection used data later than the first evaluation origin.")
    if any(
        left + timedelta(days=horizon) > right
        for left, right in zip(cutoffs, cutoffs[1:], strict=False)
    ):
        raise ValueError("Validation windows must not overlap.")
    folds = []
    for index, (cutoff, season, months) in enumerate(
        zip(cutoffs, SEASONS, SEASON_MONTHS, strict=True), 1
    ):
        fold = Fold(index, cutoff, horizon, season, "holdout" if index == 4 else "tuning")
        if fold.validation_start.month not in months:
            raise ValueError(f"Fold {index} does not start in the declared {season} season.")
        folds.append(fold)
    return folds


def validate_history(history: pd.DataFrame) -> pd.DataFrame:
    required = {"unique_id", "ds", "y", "is_available"}
    if required - set(history.columns):
        raise ValueError(f"History is missing columns: {sorted(required - set(history.columns))}")
    history = history.copy()
    history["ds"] = pd.to_datetime(history["ds"], errors="raise")
    if history.empty or history[["unique_id", "ds", "is_available"]].isna().any().any():
        raise ValueError("History must be nonempty with complete keys and availability flags.")
    if history["ds"].dt.tz is not None or not history["ds"].eq(history["ds"].dt.normalize()).all():
        raise ValueError("Sales dates must be timezone-naive calendar dates.")
    if not history["unique_id"].map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("Series identifiers must be nonempty strings.")
    if not history["is_available"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("Availability flags must be boolean.")
    history["y"] = pd.to_numeric(history["y"], errors="raise").astype("float64")
    if not np.isfinite(history["y"]).all() or history["y"].lt(0).any():
        raise ValueError("Sales must be finite and nonnegative.")
    if history.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Duplicate series/date observations.")
    history = history.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    differences = history.groupby("unique_id")["ds"].diff().dropna()
    if not differences.dt.days.eq(1).all():
        raise ValueError("Every series needs a complete daily grid; missing sales are not zero.")
    if (history["y"].gt(0) & ~history["is_available"]).any():
        raise ValueError("Positive sales cannot have an unlisted availability flag.")
    return history


def split_fold(
    history: pd.DataFrame,
    fold: Fold,
    min_train_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = history.loc[history["ds"] <= pd.Timestamp(fold.origin)].copy()
    actual = history.loc[
        history["ds"].between(
            pd.Timestamp(fold.validation_start), pd.Timestamp(fold.validation_end)
        )
    ].copy()
    series = set(history["unique_id"])
    if set(train["unique_id"]) != series or set(actual["unique_id"]) != series:
        raise ValueError(f"Fold {fold.fold_id} does not cover every selected series.")
    if train.groupby("unique_id").size().lt(min_train_days).any():
        raise ValueError(
            f"Fold {fold.fold_id} needs at least {min_train_days} training days per series."
        )
    if (
        not train.groupby("unique_id")["ds"].max().eq(pd.Timestamp(fold.origin)).all()
        or not actual.groupby("unique_id").size().eq(fold.horizon).all()
    ):
        raise ValueError(f"Fold {fold.fold_id} requires a complete origin and validation horizon.")
    return train, actual


def training_scales(train: pd.DataFrame, season_length: int) -> pd.DataFrame:
    """Seasonal RMSSE scale uses training differences, including recorded zeros."""
    if season_length < 1:
        raise ValueError("Season length must be positive.")
    train = train.sort_values(["unique_id", "ds"]).copy()
    train["seasonal_squared_difference"] = (
        train.groupby("unique_id")["y"].diff(season_length).pow(2)
    )
    scales = (
        train.groupby("unique_id")
        .agg(
            rmsse_scale=("seasonal_squared_difference", "mean"),
            training_days=("y", "size"),
            training_start=("ds", "min"),
            training_end=("ds", "max"),
            listed_training_days=("is_available", "sum"),
        )
        .reset_index()
    )
    scales["unlisted_training_days"] = scales["training_days"] - scales["listed_training_days"]
    return scales
