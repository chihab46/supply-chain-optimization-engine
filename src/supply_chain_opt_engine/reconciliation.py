"""Reconcile hierarchy points and joint paths with training-only covariance.

Run ``python -m supply_chain_opt_engine.reconciliation --source ingestion`` after
the global forecasting step. Marginal quantiles are not asserted to be additive.
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

from supply_chain_opt_engine.feature_store import load_feature_store
from supply_chain_opt_engine.hierarchy import LEVELS, Hierarchy
from supply_chain_opt_engine.m5 import checksum
from supply_chain_opt_engine.metrics import (
    QUANTILES,
    aggregate_scores,
    quantile_column,
    score_predictions,
)
from supply_chain_opt_engine.reconcilers import (
    METHODS,
    base_sample_paths,
    fit_reconcilers,
    rank_block_uniforms,
    reconcile_values,
)
from supply_chain_opt_engine.reconciliation_data import prepare_reconciliation_data

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReconciliationConfig:
    num_samples: int = 501
    ridge: float = 2e-8
    atol: float = 1e-8
    seed: int = 42
    primary_method: str = "mint_shrink"

    def __post_init__(self):
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Sampling seed must be a nonnegative integer.")
        if (
            isinstance(self.num_samples, bool)
            or not isinstance(self.num_samples, int)
            or self.num_samples < 101
            or (self.num_samples - 1) % 100 != 0
        ):
            raise ValueError("Use 101, 201, 501, ... paths to represent all quantile anchors.")
        if self.ridge <= 0 or self.atol <= 0 or not np.isfinite([self.ridge, self.atol]).all():
            raise ValueError(
                "Covariance ridge and coherence tolerance must be finite and positive."
            )
        if self.primary_method not in METHODS:
            raise ValueError("Choose a declared reconciliation method before viewing results.")


def load_forecast_bundle(root: Path, path: Path | None = None) -> tuple[Path, dict]:
    if path is None:
        candidates = list((root / "artifacts/models/lightgbm").glob("*/manifest.json"))
        if not candidates:
            raise ValueError("Complete global LightGBM forecasting before reconciliation.")
        path = max(candidates, key=lambda p: p.stat().st_mtime)
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "forecasted":
        raise ValueError("Upstream forecast run is not complete.")
    for name, digest in manifest["artifact_sha256"].items():
        if checksum(path.parent / name) != digest:
            raise ValueError(f"Upstream forecast checksum mismatch: {name}")
    manifest["_categories"] = json.loads((path.parent / "holdout/model.json").read_text())[
        "categories"
    ]
    return path, manifest


def prediction_rows(
    hierarchy: Hierarchy,
    point: np.ndarray,
    quantiles: np.ndarray,
    dates,
    origin,
    model,
    fold_id=None,
):
    rows = pd.DataFrame(
        {
            "unique_id": np.repeat(hierarchy.nodes.unique_id.to_numpy(), len(dates)),
            "ds": np.tile(dates, len(hierarchy.nodes)),
            "point": point.reshape(-1),
            "forecast_origin": pd.Timestamp(origin),
            "model": model,
        }
    )
    rows["horizon"] = (rows.ds - pd.Timestamp(origin)).dt.days
    for index, q in enumerate(QUANTILES):
        rows[quantile_column(q)] = quantiles[:, :, index].reshape(-1)
    if fold_id is not None:
        rows["fold_id"] = fold_id
    return rows


def hierarchy_scales(hierarchy: Hierarchy, actual: np.ndarray, dates, origin, fold_id):
    mask = dates <= pd.Timestamp(origin)
    train = actual[:, mask]
    if train.shape[1] <= 7:
        raise ValueError("Seasonal scaling needs more than seven training observations.")
    return pd.DataFrame(
        {
            "unique_id": hierarchy.nodes.unique_id,
            "fold_id": fold_id,
            "rmsse_scale": np.mean((train[:, 7:] - train[:, :-7]) ** 2, axis=1),
            "training_days": train.shape[1],
        }
    )


def reconcile_origin(
    hierarchy: Hierarchy,
    data,
    config: ReconciliationConfig,
    seed: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], object, list[dict]]:
    fitted = fit_reconcilers(
        hierarchy, data.base_point, data.actual_insample, data.fitted_insample, config.ridge
    )
    uniforms = rank_block_uniforms(
        data.actual_insample - data.fitted_insample,
        data.base_point.shape[1],
        config.num_samples,
        seed,
    )
    base_paths = base_sample_paths(data.base_quantiles, uniforms)
    results = {}
    diagnostics = []
    for method in METHODS:
        point, point_clips = reconcile_values(
            hierarchy, fitted.projections[method], data.base_point
        )
        samples, sample_clips = reconcile_values(hierarchy, fitted.projections[method], base_paths)
        hierarchy.assert_coherent(point, config.atol)
        hierarchy.assert_coherent(samples, config.atol)
        quantiles = np.moveaxis(np.quantile(samples, QUANTILES, axis=2), 0, -1)
        results[method] = (point, quantiles, samples)
        diagnostics.append(
            {
                "model": method,
                "point_coherence_error": hierarchy.coherence_error(point),
                "sample_coherence_error": hierarchy.coherence_error(samples),
                "marginal_q50_additivity_error": hierarchy.coherence_error(quantiles[:, :, 2]),
                "negative_point_leaves_clipped": point_clips,
                "negative_sample_leaves_clipped": sample_clips,
                "residual_observations": len(data.residual_dates),
                "residual_start": str(data.residual_dates.min().date()),
                "residual_end": str(data.residual_dates.max().date()),
                "top_down_uniform_fallback": fitted.top_down_uniform_fallback,
                "mint_ols_fallback": fitted.mint_ols_fallback,
                "aggregate_interval_fallback_cells": data.aggregate_interval_fallback_cells,
            }
        )
    return results, fitted, diagnostics


def run_reconciliation(
    store, upstream_path: Path, upstream: dict, output_root: Path, config: ReconciliationConfig
) -> Path:
    if store.provenance["ingestion_run_id"] != upstream["source"]["ingestion_run_id"]:
        raise ValueError("Forecasts and reconciliation history use different ingestion scopes.")
    catalog = (
        store.history[["unique_id", "store_id", "item_id"]]
        .drop_duplicates()
        .merge(store.items, on="item_id", validate="many_to_one")
    )
    hierarchy = Hierarchy.from_catalog(catalog)
    all_actual, dates = hierarchy.history_matrix(store.history)
    historical = pd.read_parquet(upstream_path.parent / "predictions.parquet")
    latest = pd.read_parquet(upstream_path.parent / "latest_forecasts.parquet")
    for frame in [historical, latest]:
        if set(frame.unique_id) != set(hierarchy.bottom_ids):
            raise ValueError("Upstream leaf IDs do not match the hierarchy.")
    verified = historical.merge(
        store.history[["unique_id", "ds", "y"]],
        on=["unique_id", "ds"],
        suffixes=("_forecast", "_actual"),
        validate="many_to_one",
    )
    if not np.array_equal(verified.y_forecast, verified.y_actual):
        raise ValueError("Upstream target labels differ from current sales history.")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output_root.mkdir(parents=True, exist_ok=True)
    bundle = output_root / run_id
    predictions = []
    scale_frames = []
    coherence = []
    runtimes = []
    calibration = []
    with tempfile.TemporaryDirectory(prefix=".reconciliation-", dir=output_root) as temporary:
        staging = Path(temporary)
        hierarchy.nodes.to_csv(staging / "hierarchy_nodes.csv", index=False)
        np.savez_compressed(
            staging / "hierarchy.npz",
            S=hierarchy.S,
            node_ids=hierarchy.nodes.unique_id.to_numpy(dtype=str),
            bottom_ids=np.array(hierarchy.bottom_ids, dtype=str),
        )
        (staging / "tags.json").write_text(
            json.dumps(
                {k: hierarchy.nodes.unique_id.iloc[v].tolist() for k, v in hierarchy.tags.items()},
                indent=2,
            )
            + "\n"
        )
        runs = [
            (
                fold["fold_id"],
                date.fromisoformat(fold["origin"]),
                historical[historical.fold_id.eq(fold["fold_id"])],
                upstream_path.parent / "holdout" if fold["role"] == "holdout" else None,
            )
            for fold in upstream["folds"]
        ]
        runs.append(
            (
                None,
                date.fromisoformat(upstream["latest_origin"]),
                latest,
                upstream_path.parent / "latest",
            )
        )
        for fold_id, origin, base, model_directory in runs:
            begin = time.perf_counter()
            role = f"fold_{fold_id}" if fold_id is not None else "latest"
            LOGGER.info(
                "Preparing %s: %d hierarchy nodes at origin %s", role, len(hierarchy.nodes), origin
            )
            data = prepare_reconciliation_data(
                store, hierarchy, origin, base, upstream, model_directory
            )
            results, fit, checks = reconcile_origin(
                hierarchy, data, config, config.seed + (fold_id or 100)
            )
            rows = [
                prediction_rows(
                    hierarchy,
                    data.base_point,
                    data.base_quantiles,
                    data.future_dates,
                    origin,
                    "base",
                    fold_id,
                )
            ]
            for method, (point, quantiles, samples) in results.items():
                rows.append(
                    prediction_rows(
                        hierarchy, point, quantiles, data.future_dates, origin, method, fold_id
                    )
                )
                if fold_id == 4 or fold_id is None:
                    np.savez_compressed(
                        staging / f"{role}_{method}_samples.npz",
                        samples=samples,
                        node_ids=hierarchy.nodes.unique_id.to_numpy(dtype=str),
                        ds=data.future_dates.to_numpy(dtype="datetime64[D]"),
                    )
            combined = pd.concat(rows, ignore_index=True)
            np.savez_compressed(
                staging / f"{role}_fit.npz",
                covariance=fit.covariance,
                **{f"P_{name}": P for name, P in fit.projections.items()},
                residuals=data.actual_insample - data.fitted_insample,
                residual_dates=data.residual_dates.to_numpy(dtype="datetime64[D]"),
            )
            calibration.append(
                {
                    "phase": role,
                    "origin": origin.isoformat(),
                    "source": data.source,
                    "residual_end": str(data.residual_dates.max().date()),
                    "aggregate_interval_fallback_cells": data.aggregate_interval_fallback_cells,
                }
            )
            if fold_id is not None:
                actual = all_actual[:, dates.get_indexer(data.future_dates)]
                labels = pd.DataFrame(
                    {
                        "unique_id": np.repeat(
                            hierarchy.nodes.unique_id.to_numpy(), len(data.future_dates)
                        ),
                        "ds": np.tile(data.future_dates, len(hierarchy.nodes)),
                        "y": actual.reshape(-1),
                    }
                )
                predictions.append(
                    combined.merge(labels, on=["unique_id", "ds"], validate="many_to_one")
                )
                scale_frames.append(hierarchy_scales(hierarchy, all_actual, dates, origin, fold_id))
            else:
                combined.to_parquet(staging / "latest_forecasts.parquet", index=False)
                leaf = combined[
                    combined.model.eq(config.primary_method)
                    & combined.unique_id.isin(hierarchy.bottom_ids)
                ]
                leaf.to_parquet(staging / "latest_leaf_forecasts.parquet", index=False)
            coherence.extend(
                [
                    {**check, "phase": role, "forecast_origin": origin.isoformat()}
                    for check in checks
                ]
            )
            runtimes.append({"phase": role, "seconds": time.perf_counter() - begin})
            LOGGER.info(
                "%s finished in %.2fs; max point error %.3g, sample error %.3g",
                role,
                time.perf_counter() - begin,
                max(c["point_coherence_error"] for c in checks),
                max(c["sample_coherence_error"] for c in checks),
            )
        forecast = pd.concat(predictions, ignore_index=True)
        scale = pd.concat(scale_frames, ignore_index=True)
        series = score_predictions(forecast, scale).merge(
            hierarchy.nodes[["unique_id", "level"]], on="unique_id", validate="many_to_one"
        )
        fold_metadata = pd.DataFrame(upstream["folds"])
        series = series.merge(fold_metadata, on="fold_id", validate="many_to_one")
        fold_metrics = aggregate_scores(series, ["fold_id", "level", "model"]).merge(
            fold_metadata, on="fold_id", validate="many_to_one"
        )
        summaries = []
        for scope, subset in [
            ("all_folds", series),
            ("tuning_folds", series[series.role.eq("tuning")]),
            ("holdout", series[series.role.eq("holdout")]),
        ]:
            summaries.append(aggregate_scores(subset, ["level", "model"]).assign(scope=scope))
        summary = pd.concat(summaries, ignore_index=True)
        forecast.to_parquet(staging / "predictions.parquet", index=False)
        series.to_parquet(staging / "series_metrics.parquet", index=False)
        fold_metrics.to_csv(staging / "fold_metrics.csv", index=False)
        summary.to_csv(staging / "summary_metrics.csv", index=False)
        pd.DataFrame(coherence).to_csv(staging / "coherence.csv", index=False)
        pd.DataFrame(runtimes).to_csv(staging / "runtimes.csv", index=False)
        manifest = {
            "run_id": run_id,
            "status": "reconciled",
            "config": asdict(config),
            "source": store.provenance,
            "upstream_run_id": upstream["run_id"],
            "upstream_manifest_sha256": checksum(upstream_path),
            "nodes": len(hierarchy.nodes),
            "bottom_series": hierarchy.n_bottom,
            "levels": hierarchy.nodes.level.value_counts().to_dict(),
            "methods": list(METHODS),
            "folds": upstream["folds"],
            "latest_origin": upstream["latest_origin"],
            "calibration": calibration,
            "point_policy": "Coherent projected centers, not necessarily marginal medians or means",
            "sample_policy": "Rank-stratified circular copula; exact base quantile anchors",
            "tail_policy": "P00=0; P100=P95+(P95-P90), piecewise-linear inverse CDF",
            "quantile_policy": "Marginal sample quantiles are ordered but are not claimed to add",
            "negative_policy": "Clip negative leaves, then reaggregate; not constrained MinT",
            "aggregate_base_model": "Independent AutoETS weekly model at each aggregate node",
            "aggregate_interval_policy": (
                "Native ETS; undefined cells use centered training-error quantiles"
            ),
            "covariance_policy": "MinTrace mint_shrink on aligned pre-origin fitted residuals",
            "top_down_policy": "Training-average proportions; uniform fallback at zero history",
            "mint_fallback_policy": "Explicit OLS if Nixtla rejects near-zero residual sums",
            "versions": {
                name: version(name)
                for name in ["hierarchicalforecast", "statsforecast", "numpy", "scipy", "pandas"]
            },
            "code_sha256": {
                name: checksum(Path(__file__).parent / name)
                for name in [
                    "reconciliation.py",
                    "reconciliation_data.py",
                    "hierarchy.py",
                    "reconcilers.py",
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
    print(
        summary[summary.scope.eq("holdout")][["level", "model", "wmape", "coverage_90"]].to_string(
            index=False
        )
    )
    return bundle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--source", choices=["postgres", "ingestion"], default="postgres")
    parser.add_argument("--forecast-manifest", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    engine = None
    try:
        document = yaml.safe_load(args.config.read_text())
        root = args.config.resolve().parent.parent
        options = document["reconciliation"]
        if tuple(options["methods"]) != METHODS:
            raise ValueError("Run Bottom-Up, Top-Down and MinT-shrink together.")
        if tuple(options["hierarchy"]) != LEVELS:
            raise ValueError("Use Total -> Store -> Store-Category -> Store-Item hierarchy.")
        config = ReconciliationConfig(
            num_samples=options.get("num_samples", 501),
            ridge=options.get("covariance_ridge", 2e-8),
            atol=options.get("coherence_atol", 1e-8),
            seed=document["seed"],
            primary_method=options.get("primary_method", "mint_shrink"),
        )
        path, upstream = load_forecast_bundle(root, args.forecast_manifest)
        if args.source == "postgres":
            load_dotenv(root / ".env", override=False)
            url = os.environ.get("DATABASE_URL")
            if not url:
                raise ValueError("Set DATABASE_URL or use --source ingestion.")
            engine = create_engine(url, connect_args={"connect_timeout": 10}, hide_parameters=True)
        store = load_feature_store(root, engine)
        bundle = run_reconciliation(
            store,
            path,
            upstream,
            root / document["artifacts"]["root_dir"] / "forecasts/reconciliation",
            config,
        )
        print(bundle / "manifest.json")
    except SQLAlchemyError:
        LOGGER.error("PostgreSQL snapshot unavailable. Check access or use --source ingestion.")
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
