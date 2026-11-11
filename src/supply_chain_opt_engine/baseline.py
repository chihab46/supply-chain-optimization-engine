"""Seasonal statistical backtests with training-only probabilistic calibration.

Run ``python -m supply_chain_opt_engine.baseline --source ingestion`` to use the
immutable CSVs from a committed ingestion run, or omit the flag for PostgreSQL.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from statsforecast import StatsForecast
from statsforecast.models import TSB, AutoETS, SeasonalNaive
from statsforecast.utils import ConformalIntervals

from supply_chain_opt_engine.evaluation import (
    Fold,
    seasonal_folds,
    split_fold,
    training_scales,
    validate_history,
)
from supply_chain_opt_engine.history import load_history
from supply_chain_opt_engine.m5 import checksum
from supply_chain_opt_engine.metrics import (
    QUANTILES,
    aggregate_scores,
    quantile_column,
    score_predictions,
)

LOGGER = logging.getLogger(__name__)
MODELS = ("SeasonalNaive", "AutoETS", "TSB")
INTERVAL_SUFFIX = {0.05: "lo-90", 0.1: "lo-80", 0.8: "hi-60", 0.9: "hi-80", 0.95: "hi-90"}


@dataclass(frozen=True)
class BaselineConfig:
    horizon: int = 28
    season_length: int = 7
    conformal_windows: int = 2
    tsb_alpha_d: float = 0.1
    tsb_alpha_p: float = 0.1
    n_jobs: int = 1

    def __post_init__(self) -> None:
        for name in ["horizon", "season_length", "conformal_windows", "n_jobs"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.conformal_windows < 2:
            raise ValueError("Conformal calibration requires at least two training windows.")
        if not 0 < self.tsb_alpha_d <= 1 or not 0 < self.tsb_alpha_p <= 1:
            raise ValueError("TSB smoothing parameters must lie in (0, 1].")

    @property
    def min_train_days(self) -> int:
        return self.conformal_windows * self.horizon + 3 * self.season_length


def make_model(name: str, config: BaselineConfig):
    intervals = ConformalIntervals(n_windows=config.conformal_windows, h=config.horizon)
    if name == "SeasonalNaive":
        return SeasonalNaive(season_length=config.season_length, prediction_intervals=intervals)
    if name == "AutoETS":
        return AutoETS(
            season_length=config.season_length, model="ZZZ", prediction_intervals=intervals
        )
    if name == "TSB":
        return TSB(
            alpha_d=config.tsb_alpha_d, alpha_p=config.tsb_alpha_p, prediction_intervals=intervals
        )
    raise ValueError(f"Unsupported statistical baseline: {name}")


def forecast_fold(
    history: pd.DataFrame,
    fold: Fold,
    config: BaselineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Fit each model only on observations through this fold's origin."""
    if fold.horizon != config.horizon:
        raise ValueError("Fold and model horizons disagree.")
    train, actual = split_fold(history, fold, config.min_train_days)
    training = train[["unique_id", "ds", "y"]]
    scales = training_scales(train, config.season_length).assign(fold_id=fold.fold_id)
    predictions, runtimes = [], []
    for name in MODELS:
        LOGGER.info(
            "Fold %d, %s: fitting %d series through %s",
            fold.fold_id,
            name,
            training["unique_id"].nunique(),
            fold.origin,
        )
        begin = time.perf_counter()
        forecaster = StatsForecast(
            models=[make_model(name, config)], freq="D", n_jobs=config.n_jobs
        )
        forecasts = forecaster.forecast(df=training, h=config.horizon, level=[60, 80, 90])
        elapsed = time.perf_counter() - begin
        if forecasts.duplicated(["unique_id", "ds"]).any():
            raise ValueError(f"{name} returned duplicate forecasts.")
        result = actual[["unique_id", "ds", "y", "is_available"]].merge(
            forecasts,
            on=["unique_id", "ds"],
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not result["_merge"].eq("both").all():
            raise ValueError(f"{name} forecasts do not match all validation dates and series.")
        values = result[[name, *[f"{name}-{suffix}" for suffix in INTERVAL_SUFFIX.values()]]]
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{name} returned nonfinite predictions; no fallback is applied.")
        tidy = result[["unique_id", "ds", "y", "is_available"]].copy()
        tidy["point"] = result[name].clip(lower=0)
        tidy["point_clipped"] = result[name].lt(0)
        tidy["quantile_clipped"] = values.lt(0).any(axis=1)
        for q in QUANTILES:
            column = name if q == 0.5 else f"{name}-{INTERVAL_SUFFIX[q]}"
            tidy[quantile_column(q)] = result[column].clip(lower=0)
        tidy["model"] = name
        tidy["fold_id"] = fold.fold_id
        tidy["forecast_origin"] = pd.Timestamp(fold.origin)
        tidy["horizon"] = (tidy["ds"] - pd.Timestamp(fold.origin)).dt.days
        predictions.append(tidy)
        runtimes.append(
            {"fold_id": fold.fold_id, "model": name, "fit_calibrate_predict_seconds": elapsed}
        )
        LOGGER.info("Fold %d, %s completed in %.2fs", fold.fold_id, name, elapsed)
    return pd.concat(predictions, ignore_index=True), scales, runtimes


def run_backtest(
    history: pd.DataFrame,
    folds: list[Fold],
    config: BaselineConfig,
) -> dict[str, pd.DataFrame]:
    if len(folds) != 4 or folds != seasonal_folds(
        [fold.origin for fold in folds],
        config.horizon,
        folds[0].origin,
    ):
        raise ValueError("Use four seasonal folds with fold 4 reserved as holdout.")
    history = validate_history(history)
    # Validate all folds before fitting anything, including the untouched holdout.
    for fold in folds:
        split_fold(history, fold, config.min_train_days)
    forecasts, scales, runtimes = [], [], []
    for fold in folds:
        prediction, scale, timing = forecast_fold(history, fold, config)
        forecasts.append(prediction)
        scales.append(scale)
        runtimes.extend(timing)
    predictions = pd.concat(forecasts, ignore_index=True)
    training = pd.concat(scales, ignore_index=True)
    series = score_predictions(predictions, training)
    metadata = pd.DataFrame([fold.metadata() for fold in folds])
    series = series.merge(metadata, on="fold_id", validate="many_to_one")
    fold_metrics = aggregate_scores(series, ["fold_id", "model"]).merge(
        metadata,
        on="fold_id",
        validate="many_to_one",
    )
    summaries = []
    for name, subset in [
        ("all_folds", series),
        ("tuning_folds", series[series["role"] == "tuning"]),
        ("holdout", series[series["role"] == "holdout"]),
    ]:
        if not subset.empty:
            summaries.append(aggregate_scores(subset, ["model"]).assign(scope=name))
    return {
        "predictions": predictions,
        "training_scales": training,
        "series_metrics": series,
        "fold_metrics": fold_metrics,
        "summary_metrics": pd.concat(summaries, ignore_index=True),
        "runtimes": pd.DataFrame(runtimes),
    }


def save_backtest(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    folds: list[Fold],
    config: BaselineConfig,
    provenance: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    bundle = output_dir / run_id
    with tempfile.TemporaryDirectory(prefix=".baseline-", dir=output_dir) as temporary:
        staging = Path(temporary)
        artifacts = []
        for name, frame in tables.items():
            extension = (
                "csv" if name in ["fold_metrics", "summary_metrics", "runtimes"] else "parquet"
            )
            path = staging / f"{name}.{extension}"
            if extension == "csv":
                frame.to_csv(path, index=False)
            else:
                frame.to_parquet(path, index=False)
            artifacts.append(path.name)
        manifest = {
            "run_id": run_id,
            "status": "backtested",
            "config": asdict(config),
            "folds": [fold.metadata() for fold in folds],
            "source": provenance,
            "models": list(MODELS),
            "quantiles": list(QUANTILES),
            "prediction_rows": len(tables["predictions"]),
            "point_values_clipped": int(tables["predictions"]["point_clipped"].sum()),
            "rows_with_quantiles_clipped": int(tables["predictions"]["quantile_clipped"].sum()),
            "interval_method": "StatsForecast conformal_distribution, nested training-only windows",
            "sales_policy": "Recorded sales include unlisted zeros; no demand imputation.",
            "rmsse_policy": "Training seasonal scale; equal weight per defined series-fold RMSSE.",
            "wmape_policy": "Pooled absolute errors / actual units; undefined at zero demand.",
            "probability_policy": "Observation-weighted pinball; inclusive P05-P95 coverage.",
            "summary_policy": "Separate all-fold, tuning-only and untouched holdout summaries.",
            "versions": {
                name: version(name) for name in ["statsforecast", "numpy", "pandas", "scipy"]
            },
            "code_sha256": {
                name: checksum(Path(__file__).parent / name)
                for name in ["baseline.py", "evaluation.py", "metrics.py", "history.py"]
            },
            "artifact_sha256": {name: checksum(staging / name) for name in artifacts},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(bundle)
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--source", choices=["postgres", "ingestion"], default="postgres")
    parser.add_argument(
        "--ingestion-manifest", type=Path, help="Defaults to the latest committed ingestion"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    engine = None
    try:
        document = yaml.safe_load(args.config.read_text())
        root = args.config.resolve().parent.parent
        options = document["backtest"]
        if (
            options["n_windows"] != 4
            or options["tuning_fold_ids"] != [1, 2, 3]
            or options["holdout_fold_id"] != 4
        ):
            raise ValueError(
                "Use four folds, with folds 1-3 for tuning and fold 4 as strict holdout."
            )
        config = BaselineConfig(
            horizon=options["horizon_days"],
            season_length=options["season_length"],
            conformal_windows=options.get("conformal_windows", 2),
            tsb_alpha_d=options.get("tsb_alpha_d", 0.1),
            tsb_alpha_p=options.get("tsb_alpha_p", 0.1),
            n_jobs=options.get("n_jobs", 1),
        )
        if args.source == "postgres":
            load_dotenv(root / ".env", override=False)
            url = os.environ.get("DATABASE_URL")
            if not url:
                raise ValueError("Set DATABASE_URL or use --source ingestion for committed CSVs.")
            engine = create_engine(url, connect_args={"connect_timeout": 10}, hide_parameters=True)
        history, provenance = load_history(root, engine, args.ingestion_manifest)
        cutoffs = [date.fromisoformat(str(value)) for value in options["cutoff_dates"]]
        selection_cutoff = max(
            date.fromisoformat(str(document["data"]["selection_cutoff"])),
            date.fromisoformat(provenance["selection_cutoff"]),
        )
        folds = seasonal_folds(cutoffs, config.horizon, selection_cutoff)
        tables = run_backtest(history, folds, config)
        bundle = save_backtest(
            tables,
            root / document["artifacts"]["root_dir"] / "metrics" / "baselines",
            folds,
            config,
            provenance,
        )
        LOGGER.info("Saved statistical benchmark at %s", bundle)
        print(
            tables["summary_metrics"][
                ["scope", "model", "wmape", "rmsse", "coverage_90", "pinball_mean"]
            ].to_string(index=False)
        )
        print(bundle / "manifest.json")
    except SQLAlchemyError:
        LOGGER.error("Could not read PostgreSQL. Check database access or use --source ingestion.")
        return 1
    except (ValueError, OSError, KeyError, ImportError, yaml.YAMLError) as error:
        LOGGER.error("%s", error)
        return 1
    finally:
        if engine is not None:
            engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
