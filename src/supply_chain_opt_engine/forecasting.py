"""Global LightGBM quantiles, tuning-only Optuna search and frozen holdout scores.

Run ``python -m supply_chain_opt_engine.forecasting --source ingestion`` to use
checksummed committed extracts without requiring a live PostgreSQL connection.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sqlite3
import sys
import tempfile
import time
import uuid
from dataclasses import asdict
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from supply_chain_opt_engine.evaluation import seasonal_folds, split_fold, training_scales
from supply_chain_opt_engine.explanations import explain_median
from supply_chain_opt_engine.feature_store import FeatureEncoder, FeatureStore, load_feature_store
from supply_chain_opt_engine.learning import (
    ForecastConfig,
    fit_quantiles,
    forecast_frame,
    predict_raw,
    prediction_dataset,
    save_models,
    training_dataset,
    training_origins,
)
from supply_chain_opt_engine.m5 import checksum
from supply_chain_opt_engine.metrics import QUANTILES, aggregate_scores, score_predictions
from supply_chain_opt_engine.tuning import FoldData, tune_parameters

LOGGER = logging.getLogger(__name__)


def config_from_document(document: dict) -> ForecastConfig:
    options = document["forecast"]
    if tuple(options["quantiles"]) != QUANTILES:
        raise ValueError("Use the shared P05/P10/P50/P80/P90/P95 quantile contract.")
    if document["backtest"]["horizon_days"] != options["horizon_days"]:
        raise ValueError("Forecast and backtest horizons must agree.")
    if (
        document["backtest"]["tuning_fold_ids"] != [1, 2, 3]
        or document["backtest"]["holdout_fold_id"] != 4
        or document["backtest"]["n_windows"] != 4
    ):
        raise ValueError("Reserve fold 4 as holdout and use folds 1-3 for tuning.")
    return ForecastConfig(
        horizon=options["horizon_days"],
        training_window_days=options.get("training_window_days", 365),
        origin_stride_days=options.get("origin_stride_days", 28),
        n_estimators=options.get("n_estimators", 160),
        optuna_trials=options["optuna_trials"],
        num_threads=options.get("num_threads", 2),
        seed=document["seed"],
        shap_rows=options.get("shap_rows", 600),
    )


def run_forecasting(store: FeatureStore, document: dict, output_root: Path) -> Path:
    config = config_from_document(document)
    selection_cutoff = max(
        date.fromisoformat(str(document["data"]["selection_cutoff"])),
        date.fromisoformat(store.provenance["selection_cutoff"]),
    )
    folds = seasonal_folds(
        [date.fromisoformat(str(value)) for value in document["backtest"]["cutoff_dates"]],
        config.horizon,
        selection_cutoff,
    )
    for fold in folds:
        split_fold(store.history, fold, min_train_days=29 + 2 * config.horizon)
    encoder = FeatureEncoder.from_store(store)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output_root.mkdir(parents=True, exist_ok=True)
    study_path = output_root / f"{run_id}.sqlite3"
    tuning_data = []
    for fold in folds[:3]:
        LOGGER.info("Preparing tuning fold %d through %s", fold.fold_id, fold.origin)
        tuning_data.append(
            FoldData(
                fold,
                training_dataset(store, encoder, fold.origin, config),
                prediction_dataset(store, encoder, fold.origin, config, with_labels=True),
            )
        )
    study = tune_parameters(tuning_data, config, study_path)
    best_params = study.best_params
    LOGGER.info("Search complete. Frozen parameters: %s", best_params)
    predictions, scales, timings, origin_metadata = [], [], [], []
    bundle = output_root / run_id
    with tempfile.TemporaryDirectory(prefix=".lightgbm-", dir=output_root) as temporary:
        staging = Path(temporary)
        # No holdout dataset is constructed or passed to Optuna. The fourth fold
        # is prepared only now, after parameter selection has finished.
        for index, fold in enumerate(folds):
            begin = time.perf_counter()
            if index < 3:
                data = tuning_data[index]
            else:
                data = FoldData(
                    fold,
                    training_dataset(store, encoder, fold.origin, config),
                    prediction_dataset(store, encoder, fold.origin, config, with_labels=True),
                )
            LOGGER.info("Fitting six global quantile models for fold %d", fold.fold_id)
            model = fit_quantiles(data.train, best_params, config)
            raw = predict_raw(model, data.validation.X)
            predictions.append(forecast_frame(raw, data.validation, fold.fold_id))
            train, _ = split_fold(store.history, fold, 29 + 2 * config.horizon)
            scales.append(
                training_scales(train, document["backtest"]["season_length"]).assign(
                    fold_id=fold.fold_id
                )
            )
            timings.append(
                {
                    "phase": f"fold_{fold.fold_id}",
                    "train_predict_seconds": time.perf_counter() - begin,
                    "training_rows": len(data.train.X),
                }
            )
            origin_metadata.append(
                {
                    "fold_id": fold.fold_id,
                    "training_origins": [
                        value.isoformat() for value in training_origins(store, fold.origin, config)
                    ],
                    "max_label_date": str(data.train.keys["ds"].max().date()),
                }
            )
            if fold.role == "holdout":
                save_models(
                    model,
                    encoder,
                    staging / "holdout",
                    {"origin": fold.origin, "params": best_params, "role": "frozen_holdout"},
                )
                diagnostics = explain_median(
                    model.models_["q50"], data.validation, config.shap_rows, config.seed
                )
                diagnostics.rows.to_parquet(staging / "holdout_shap.parquet", index=False)
                diagnostics.importance.to_csv(staging / "holdout_shap_importance.csv", index=False)
            LOGGER.info("Fold %d finished in %.2fs", fold.fold_id, time.perf_counter() - begin)
        forecast = pd.concat(predictions, ignore_index=True)
        scale_frame = pd.concat(scales, ignore_index=True)
        series = score_predictions(forecast, scale_frame)
        metadata = pd.DataFrame([fold.metadata() for fold in folds])
        series = series.merge(metadata, on="fold_id", validate="many_to_one")
        fold_scores = aggregate_scores(series, ["fold_id", "model"]).merge(
            metadata, on="fold_id", validate="many_to_one"
        )
        summaries = []
        for scope, subset in [
            ("all_folds", series),
            ("tuning_folds", series[series.role.eq("tuning")]),
            ("holdout", series[series.role.eq("holdout")]),
        ]:
            summaries.append(aggregate_scores(subset, ["model"]).assign(scope=scope))
        summary = pd.concat(summaries, ignore_index=True)
        forecast.to_parquet(staging / "predictions.parquet", index=False)
        scale_frame.to_parquet(staging / "training_scales.parquet", index=False)
        series.to_parquet(staging / "series_metrics.parquet", index=False)
        fold_scores.to_csv(staging / "fold_metrics.csv", index=False)
        summary.to_csv(staging / "summary_metrics.csv", index=False)
        study.trials_dataframe().to_csv(staging / "optuna_trials.csv", index=False)
        (staging / "best_params.json").write_text(json.dumps(best_params, indent=2) + "\n")
        with (
            sqlite3.connect(study_path) as source_db,
            sqlite3.connect(staging / "study.sqlite3") as destination,
        ):
            source_db.backup(destination)
        # Production refit happens after evaluation. It may use newer observations,
        # but cannot alter the saved holdout forecasts or reported metrics.
        latest = store.history["ds"].max().date()
        begin = time.perf_counter()
        LOGGER.info("Refitting frozen parameters for latest origin %s", latest)
        production_train = training_dataset(store, encoder, latest, config)
        production_data = prediction_dataset(store, encoder, latest, config)
        production = fit_quantiles(production_train, best_params, config)
        latest_raw = predict_raw(production, production_data.X)
        forecast_frame(latest_raw, production_data).to_parquet(
            staging / "latest_forecasts.parquet", index=False
        )
        save_models(
            production,
            encoder,
            staging / "latest",
            {"origin": latest, "params": best_params, "role": "production_refit"},
        )
        diagnostics = explain_median(
            production.models_["q50"], production_data, config.shap_rows, config.seed
        )
        diagnostics.rows.to_parquet(staging / "latest_shap.parquet", index=False)
        diagnostics.importance.to_csv(staging / "latest_shap_importance.csv", index=False)
        timings.append(
            {
                "phase": "production_refit",
                "train_predict_seconds": time.perf_counter() - begin,
                "training_rows": len(production_train.X),
            }
        )
        pd.DataFrame(timings).to_csv(staging / "runtimes.csv", index=False)
        manifest = {
            "run_id": run_id,
            "status": "forecasted",
            "config": asdict(config),
            "source": store.provenance,
            "folds": [fold.metadata() for fold in folds],
            "training_blocks": origin_metadata,
            "latest_origin": latest.isoformat(),
            "best_params": best_params,
            "optuna_objective": "Mean median pinball over tuning folds 1-3 only",
            "best_trial": study.best_trial.number,
            "best_tuning_value": study.best_value,
            "trial_states": study.trials_dataframe()["state"].value_counts().to_dict(),
            "quantiles": list(QUANTILES),
            "feature_names": production_train.X.columns.tolist(),
            "prediction_rows": len(forecast),
            "latest_rows": len(production_data.X),
            "repaired_rows": int(forecast["quantile_repaired"].sum()),
            "repair_policy": "Nonnegative P50-anchored sorting/clamping of other quantiles",
            "shap_policy": "TreeSHAP of raw P50 on day-one rows; no causal claims",
            "price_policy": "Carry forward origin-known prices; future prices are not inputs",
            "sales_policy": "Recorded labels retain unlisted zeros; demand statistics exclude them",
            "training_policy": "Non-overlapping horizon blocks; labels end by each cutoff",
            "rmsse_policy": "Full pre-origin seasonal scale, as in baseline scoring",
            "wmape_policy": "Pooled absolute P50 errors divided by actual units",
            "coverage_policy": "Inclusive central P05-P95 coverage, observation weighted",
            "runtime": {
                "platform": platform.platform(),
                "visible_cpus": os.cpu_count(),
                "model_threads": config.num_threads,
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
                },
            },
            "versions": {
                name: version(name)
                for name in ["lightgbm", "mlforecast", "optuna", "shap", "numpy", "pandas"]
            },
            "code_sha256": {
                name: checksum(Path(__file__).parent / name)
                for name in [
                    "forecasting.py",
                    "learning.py",
                    "tuning.py",
                    "explanations.py",
                    "feature_store.py",
                    "evaluation.py",
                    "metrics.py",
                ]
            },
            "artifact_sha256": {
                str(path.relative_to(staging)): checksum(path)
                for path in staging.rglob("*")
                if path.is_file()
            },
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        staging.rename(bundle)
    LOGGER.info("Saved forecast bundle at %s", bundle)
    print(
        summary[["scope", "wmape", "rmsse", "coverage_90", "pinball_mean"]].to_string(index=False)
    )
    return bundle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--source", choices=["postgres", "ingestion"], default="postgres")
    parser.add_argument("--ingestion-manifest", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    engine = None
    try:
        document = yaml.safe_load(args.config.read_text())
        root = args.config.resolve().parent.parent
        config_from_document(document)
        if args.source == "postgres":
            load_dotenv(root / ".env", override=False)
            url = os.environ.get("DATABASE_URL")
            if not url:
                raise ValueError("Set DATABASE_URL or use --source ingestion.")
            engine = create_engine(url, connect_args={"connect_timeout": 10}, hide_parameters=True)
        store = load_feature_store(root, engine, args.ingestion_manifest)
        output = root / document["artifacts"]["root_dir"] / "models/lightgbm"
        bundle = run_forecasting(store, document, output)
        print(bundle / "manifest.json")
    except SQLAlchemyError:
        LOGGER.error("Database feature snapshot failed. Check access or use --source ingestion.")
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
