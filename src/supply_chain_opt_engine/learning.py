"""Direct horizon-conditioned global LightGBM models fitted through MLForecast."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast

from supply_chain_opt_engine.feature_store import FeatureEncoder, FeatureStore
from supply_chain_opt_engine.metrics import QUANTILES, quantile_column


@dataclass(frozen=True)
class ForecastConfig:
    horizon: int = 28
    training_window_days: int = 365
    origin_stride_days: int = 28
    n_estimators: int = 160
    optuna_trials: int = 30
    num_threads: int = 2
    seed: int = 42
    shap_rows: int = 600

    def __post_init__(self):
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if name == "seed" else 1)
            ):
                raise ValueError(f"{name} must be a valid integer.")
        if self.horizon > 28 or self.training_window_days < 2 * self.horizon:
            raise ValueError("Use a horizon up to 28 and at least two horizons of training labels.")
        if self.origin_stride_days != self.horizon:
            raise ValueError(
                "Direct training uses non-overlapping origin blocks: stride must equal horizon."
            )


@dataclass
class DirectDataset:
    X: pd.DataFrame
    keys: pd.DataFrame
    y: np.ndarray | None = None


def training_origins(store: FeatureStore, cutoff: date, config: ForecastConfig) -> list[date]:
    first_common = store.history.groupby("unique_id")["ds"].min().max().date() + timedelta(days=28)
    first = max(first_common, cutoff - timedelta(days=config.training_window_days))
    last = cutoff - timedelta(days=config.horizon)
    count = (last - first).days // config.origin_stride_days + 1
    if count < 2:
        raise ValueError("Not enough observed history for two direct training origin blocks.")
    return [
        last - timedelta(days=index * config.origin_stride_days) for index in reversed(range(count))
    ]


def prediction_dataset(
    store: FeatureStore,
    encoder: FeatureEncoder,
    origin: date,
    config: ForecastConfig,
    with_labels: bool = False,
) -> DirectDataset:
    panel = store.panel(origin, config.horizon)
    keys = panel[["unique_id", "forecast_origin", "ds", "horizon"]].copy()
    y = None
    if with_labels:
        lookup = store.history.set_index(["unique_id", "ds"])["y"]
        index = pd.MultiIndex.from_frame(keys[["unique_id", "ds"]])
        y = lookup.reindex(index).to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Every historical prediction row needs an observed target label.")
    return DirectDataset(encoder.transform(panel), keys, y)


def training_dataset(
    store: FeatureStore,
    encoder: FeatureEncoder,
    cutoff: date,
    config: ForecastConfig,
) -> DirectDataset:
    blocks = [
        prediction_dataset(store, encoder, origin, config, with_labels=True)
        for origin in training_origins(store, cutoff, config)
    ]
    keys = pd.concat([block.keys for block in blocks], ignore_index=True)
    if keys["ds"].max() > pd.Timestamp(cutoff) or keys.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Training labels overlap or extend beyond the training cutoff.")
    return DirectDataset(
        pd.concat([block.X for block in blocks], ignore_index=True),
        keys,
        np.concatenate([block.y for block in blocks]),
    )


def fit_quantiles(
    data: DirectDataset,
    params: dict,
    config: ForecastConfig,
    quantiles: tuple[float, ...] = QUANTILES,
) -> MLForecast:
    if data.y is None or len(data.y) != len(data.X) or not np.isfinite(data.y).all():
        raise ValueError("Global training requires complete observed labels.")
    common = dict(
        n_estimators=config.n_estimators,
        objective="quantile",
        n_jobs=config.num_threads,
        random_state=config.seed,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        max_bin=63,
        reg_lambda=1.0,
        **params,
    )
    models = {quantile_column(q): lgb.LGBMRegressor(alpha=q, **common) for q in quantiles}
    forecaster = MLForecast(models=models, freq="D", num_threads=config.num_threads)
    frame = data.keys[["unique_id", "ds"]].reset_index(drop=True).copy()
    frame["y"] = data.y
    frame = pd.concat([frame, data.X.reset_index(drop=True)], axis=1)
    # Origin features are already horizon-safe. MLForecast handles panel
    # preprocessing and global fitting, with no extra recursive demand lags.
    X, y = forecaster.preprocess(frame, static_features=[], dropna=False, return_X_y=True)
    forecaster.fit_models(X, y)
    return forecaster


def predict_raw(forecaster: MLForecast, X: pd.DataFrame) -> pd.DataFrame:
    # Each model sees the horizon as an input, so all 28 days can be predicted
    # together. Calling recursive prediction would add unnecessary daily loops.
    result = pd.DataFrame({name: model.predict(X) for name, model in forecaster.models_.items()})
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("LightGBM returned nonfinite forecasts.")
    return result


def repair_quantiles(raw: pd.DataFrame) -> pd.DataFrame:
    """Order lower/upper quantiles around a fixed nonnegative P50 anchor."""
    columns = [quantile_column(q) for q in QUANTILES]
    values = raw[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Quantile forecasts must be finite.")
    median = np.maximum(values[:, 2], 0)
    lower = np.minimum(np.maximum(np.sort(values[:, :2], axis=1), 0), median[:, None])
    upper = np.maximum(np.sort(values[:, 3:], axis=1), median[:, None])
    return pd.DataFrame(np.column_stack([lower, median, upper]), columns=columns)


def forecast_frame(
    raw: pd.DataFrame, dataset: DirectDataset, fold_id: int | None = None
) -> pd.DataFrame:
    repaired = repair_quantiles(raw)
    frame = pd.concat([dataset.keys.reset_index(drop=True), repaired], axis=1)
    frame["point"] = frame["q50"]
    frame["raw_q50"] = raw["q50"].to_numpy()
    frame["quantile_repaired"] = ~np.isclose(
        raw.to_numpy(), repaired.to_numpy(), rtol=0, atol=1e-10
    ).all(axis=1)
    frame["model"] = "LightGBM"
    if dataset.y is not None:
        frame["y"] = dataset.y
    if fold_id is not None:
        frame["fold_id"] = fold_id
    return frame


def save_models(forecaster: MLForecast, encoder: FeatureEncoder, directory: Path, info: dict):
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in forecaster.models_.items():
        model.booster_.save_model(str(directory / f"{name}.txt"))
    metadata = {
        **info,
        "categories": encoder.categories,
        "feature_names": next(iter(forecaster.models_.values())).feature_name_,
    }
    (directory / "model.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")


def predict_saved(directory: Path, panel: pd.DataFrame) -> pd.DataFrame:
    metadata = json.loads((directory / "model.json").read_text())
    X = FeatureEncoder(metadata["categories"]).transform(panel)[metadata["feature_names"]]
    raw = pd.DataFrame(
        {
            quantile_column(q): lgb.Booster(
                model_file=str(directory / f"{quantile_column(q)}.txt")
            ).predict(X)
            for q in QUANTILES
        }
    )
    return repair_quantiles(raw)
