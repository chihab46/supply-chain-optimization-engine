"""Forecast scores with explicit undefined denominators and aggregation weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = (0.05, 0.10, 0.50, 0.80, 0.90, 0.95)


def quantile_column(quantile: float) -> str:
    return f"q{round(100 * quantile):02d}"


def pinball_loss(actual: np.ndarray, forecast: np.ndarray, quantile: float) -> float:
    if not 0 < quantile < 1:
        raise ValueError("Quantile must be strictly between zero and one.")
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    if actual.shape != forecast.shape or actual.size == 0:
        raise ValueError("Pinball inputs must have the same nonempty shape.")
    if not np.isfinite(actual).all() or not np.isfinite(forecast).all():
        raise ValueError("Pinball inputs must be finite.")
    errors = actual - forecast
    return float(np.maximum(quantile * errors, (quantile - 1) * errors).mean())


def score_predictions(predictions: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    """One score row per series, fold and model. WMAPE remains undefined at zero demand."""
    keys = ["fold_id", "unique_id", "model"]
    quantile_columns = [quantile_column(q) for q in QUANTILES]
    required = {*keys, "ds", "y", "point", *quantile_columns}
    if required - set(predictions):
        raise ValueError("Prediction panel is missing required score columns.")
    if predictions.duplicated([*keys, "ds"]).any():
        raise ValueError("Duplicate forecast observations.")
    matrix = predictions[["y", "point", *quantile_columns]].to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Actual demand and predictions must be finite and nonnegative.")
    if (np.diff(predictions[quantile_columns].to_numpy(), axis=1) < -1e-10).any():
        raise ValueError("Forecast quantiles cross.")
    scored = predictions.copy()
    errors = scored["y"] - scored["point"]
    scored["absolute_error"] = errors.abs()
    scored["squared_error"] = errors.pow(2)
    scored["covered_90"] = scored["y"].between(scored["q05"], scored["q95"])
    aggregations = {
        "absolute_error_sum": ("absolute_error", "sum"),
        "squared_error_sum": ("squared_error", "sum"),
        "actual_units": ("y", "sum"),
        "observations": ("y", "size"),
        "coverage_90": ("covered_90", "mean"),
    }
    for q in QUANTILES:
        name = f"pinball_{quantile_column(q)}"
        difference = scored["y"] - scored[quantile_column(q)]
        scored[name] = np.maximum(q * difference, (q - 1) * difference)
        aggregations[name] = (name, "mean")
    result = scored.groupby(keys).agg(**aggregations).reset_index()
    result = result.merge(scales, on=["fold_id", "unique_id"], how="left", validate="many_to_one")
    if result["training_days"].isna().any():
        raise ValueError("Training-only RMSSE scales are missing for a forecast series.")
    result["wmape"] = result["absolute_error_sum"] / result["actual_units"].replace(0, np.nan)
    valid_scale = result["rmsse_scale"].where(result["rmsse_scale"] > 0)
    result["rmsse"] = np.sqrt(result["squared_error_sum"] / result["observations"] / valid_scale)
    result["pinball_mean"] = result[[f"pinball_{column}" for column in quantile_columns]].mean(
        axis=1
    )
    return result


def aggregate_scores(series_scores: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Pooled WMAPE, equal-series RMSSE, observation-weighted pinball/coverage."""
    rows = []
    for key, group in series_scores.groupby(by, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        weights = group["observations"].to_numpy(dtype=float)
        demand = float(group["actual_units"].sum())
        row = dict(zip(by, key, strict=True))
        row.update(
            wmape=float(group["absolute_error_sum"].sum()) / demand if demand > 0 else np.nan,
            rmsse=float(group["rmsse"].mean()),
            actual_units=demand,
            observations=int(weights.sum()),
            series_fold_count=len(group),
            undefined_wmape_series=int(group["wmape"].isna().sum()),
            undefined_rmsse_series=int(group["rmsse"].isna().sum()),
        )
        for metric in [
            "coverage_90",
            "pinball_mean",
            *[f"pinball_{quantile_column(q)}" for q in QUANTILES],
        ]:
            row[metric] = float(np.average(group[metric], weights=weights))
        rows.append(row)
    return pd.DataFrame(rows)
