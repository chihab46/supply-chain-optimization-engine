"""Independent aggregate forecasts and aligned pre-origin residuals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from supply_chain_opt_engine.feature_store import FeatureEncoder, FeatureStore
from supply_chain_opt_engine.hierarchy import Hierarchy
from supply_chain_opt_engine.learning import (
    ForecastConfig,
    fit_quantiles,
    predict_raw,
    prediction_dataset,
    training_dataset,
)
from supply_chain_opt_engine.metrics import QUANTILES, quantile_column


@dataclass
class ReconciliationData:
    base_point: np.ndarray
    base_quantiles: np.ndarray
    actual_insample: np.ndarray
    fitted_insample: np.ndarray
    residual_dates: pd.DatetimeIndex
    future_dates: pd.DatetimeIndex
    source: str
    aggregate_interval_fallback_cells: int = 0


def prepare_reconciliation_data(
    store: FeatureStore,
    hierarchy: Hierarchy,
    origin: date,
    base_forecasts: pd.DataFrame,
    upstream: dict,
    model_directory: Path | None,
) -> ReconciliationData:
    config = ForecastConfig(**upstream["config"])
    # The fixed schema is the one used by every upstream fold and production fit.
    encoder = FeatureEncoder(upstream["_categories"])
    train = training_dataset(store, encoder, origin, config)
    future = prediction_dataset(store, encoder, origin, config)
    expected = base_forecasts.pivot(index="unique_id", columns="ds", values="point").reindex(
        hierarchy.bottom_ids
    )
    if expected.shape != (hierarchy.n_bottom, config.horizon) or expected.isna().any().any():
        raise ValueError("Upstream forecasts do not cover every hierarchy leaf and future date.")
    if model_directory is None:
        model = fit_quantiles(train, upstream["best_params"], config, quantiles=(0.5,))
        fitted = np.maximum(predict_raw(model, train.X)["q50"].to_numpy(), 0)
        check = np.maximum(predict_raw(model, future.X)["q50"].to_numpy(), 0)
        source = "deterministic_refit_of_frozen_upstream_median"
    else:
        model = lgb.Booster(model_file=str(model_directory / "q50.txt"))
        metadata = json.loads((model_directory / "model.json").read_text())
        fitted = np.maximum(
            model.predict(train.X[metadata["feature_names"]], num_threads=config.num_threads), 0
        )
        check = np.maximum(
            model.predict(future.X[metadata["feature_names"]], num_threads=config.num_threads), 0
        )
        source = "saved_upstream_median"
    reference = future.keys.merge(
        base_forecasts[["unique_id", "ds", "point"]], on=["unique_id", "ds"], validate="one_to_one"
    )
    if not np.allclose(reference.point, check, rtol=1e-6, atol=1e-6):
        raise ValueError("Recreated median model does not reproduce the upstream leaf forecasts.")
    leaf_fit = (
        train.keys.assign(fitted=fitted)
        .pivot(index="unique_id", columns="ds", values="fitted")
        .reindex(hierarchy.bottom_ids)
    )
    residual_dates = pd.DatetimeIndex(leaf_fit.columns)
    if residual_dates.max() > pd.Timestamp(origin):
        raise ValueError("Reconciliation residuals extend past the forecast origin.")
    all_actual, all_dates = hierarchy.history_matrix(store.history)
    positions = all_dates.get_indexer(residual_dates)
    if (positions < 0).any():
        raise ValueError("Training residual labels are missing from history.")
    actual = all_actual[:, positions]
    n_agg = len(hierarchy.nodes) - hierarchy.n_bottom
    training_dates = all_dates[all_dates <= pd.Timestamp(origin)]
    upper_history = pd.DataFrame(
        {
            "unique_id": np.repeat(
                hierarchy.nodes.unique_id.iloc[:n_agg].to_numpy(), len(training_dates)
            ),
            "ds": np.tile(training_dates, n_agg),
            "y": all_actual[:n_agg, : len(training_dates)].reshape(-1),
        }
    )
    forecaster = StatsForecast(models=[AutoETS(season_length=7, model="ZZZ")], freq="D", n_jobs=1)
    upper = forecaster.forecast(df=upper_history, h=config.horizon, level=[60, 80, 90], fitted=True)
    upper_fit = forecaster.forecast_fitted_values()
    upper_ids = hierarchy.nodes.unique_id.iloc[:n_agg].tolist()
    upper_point = (
        upper.pivot(index="unique_id", columns="ds", values="AutoETS").reindex(upper_ids).to_numpy()
    )
    upper_fitted = (
        upper_fit.pivot(index="unique_id", columns="ds", values="AutoETS")
        .reindex(index=upper_ids, columns=residual_dates)
        .to_numpy()
    )
    all_fitted = np.vstack([np.maximum(upper_fitted, 0), leaf_fit.to_numpy()])
    if not np.isfinite(all_fitted).all():
        raise ValueError(
            "Aggregate and leaf in-sample predictions must align on complete finite dates."
        )
    base_point = np.vstack([np.maximum(upper_point, 0), expected.to_numpy()])
    suffix = {0.05: "lo-90", 0.1: "lo-80", 0.8: "hi-60", 0.9: "hi-80", 0.95: "hi-90"}
    quantiles = []
    for q in QUANTILES:
        upper_column = "AutoETS" if q == 0.5 else f"AutoETS-{suffix[q]}"
        a = (
            upper.pivot(index="unique_id", columns="ds", values=upper_column)
            .reindex(upper_ids)
            .to_numpy()
        )
        b = (
            base_forecasts.pivot(index="unique_id", columns="ds", values=quantile_column(q))
            .reindex(hierarchy.bottom_ids)
            .to_numpy()
        )
        quantiles.append(np.vstack([np.maximum(a, 0), b]))
    base_quantiles = np.stack(quantiles, axis=2)
    bad = ~np.isfinite(base_quantiles[:n_agg]).all(axis=2)
    # Exact/near-exact ETS fits can produce a negative numerical variance and
    # undefined native intervals. Use centered, pre-origin fitted errors only
    # for those cells, and retain an explicit fallback count.
    if bad.any():
        errors = actual[:n_agg] - all_fitted[:n_agg]
        offsets = np.quantile(errors, QUANTILES, axis=1).T
        offsets = offsets - offsets[:, 2:3]
        empirical = np.maximum(base_point[:n_agg, :, None] + offsets[:, None, :], 0)
        base_quantiles[:n_agg] = np.where(bad[:, :, None], empirical, base_quantiles[:n_agg])
    # Native aggregate intervals can cross after numerical optimization. Use the
    # same fixed median-anchored repair rule as the leaf forecasting stage.
    median = base_quantiles[:, :, 2]
    base_quantiles[:, :, :2] = np.minimum(
        np.sort(base_quantiles[:, :, :2], axis=2), median[:, :, None]
    )
    base_quantiles[:, :, 3:] = np.maximum(
        np.sort(base_quantiles[:, :, 3:], axis=2), median[:, :, None]
    )
    return ReconciliationData(
        base_point,
        base_quantiles,
        actual,
        all_fitted,
        residual_dates,
        pd.DatetimeIndex(expected.columns),
        source,
        int(bad.sum()),
    )
