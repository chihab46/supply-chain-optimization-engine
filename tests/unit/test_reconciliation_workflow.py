from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from supply_chain_opt_engine.feature_store import FeatureStore
from supply_chain_opt_engine.forecasting import run_forecasting
from supply_chain_opt_engine.hierarchy import Hierarchy
from supply_chain_opt_engine.m5 import checksum
from supply_chain_opt_engine.reconcilers import fit_reconcilers
from supply_chain_opt_engine.reconciliation import (
    ReconciliationConfig,
    load_forecast_bundle,
    run_reconciliation,
)
from supply_chain_opt_engine.reconciliation_data import prepare_reconciliation_data


def test_reconciliation_workflow_and_training_only_residuals(model_inputs, tmp_path):
    provenance = {
        "source_type": "test",
        "selection_cutoff": "2019-01-01",
        "ingestion_run_id": "unit",
    }
    store = FeatureStore.from_frames(*model_inputs, provenance=provenance)
    document = yaml.safe_load(Path("configs/default.yaml").read_text())
    document["data"]["selection_cutoff"] = "2019-01-01"
    document["backtest"].update(
        horizon_days=7, cutoff_dates=["2020-02-29", "2020-05-31", "2020-08-16", "2020-11-29"]
    )
    document["forecast"].update(
        horizon_days=7,
        origin_stride_days=7,
        training_window_days=28,
        n_estimators=6,
        optuna_trials=1,
        num_threads=1,
        shap_rows=2,
    )
    forecast = run_forecasting(store, document, tmp_path / "global")
    path, upstream = load_forecast_bundle(tmp_path, forecast / "manifest.json")
    bundle = run_reconciliation(
        store, path, upstream, tmp_path / "reconciled", ReconciliationConfig(num_samples=101)
    )
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["nodes"] == 7 and manifest["bottom_series"] == 2
    for name, digest in manifest["artifact_sha256"].items():
        assert checksum(bundle / name) == digest
    checks = pd.read_csv(bundle / "coherence.csv")
    assert checks.point_coherence_error.max() < 1e-8
    assert checks.sample_coherence_error.max() < 1e-8
    nodes = pd.read_csv(bundle / "hierarchy_nodes.csv")
    for method in ["bottom_up", "top_down", "mint_shrink"]:
        data = np.load(bundle / f"latest_{method}_samples.npz", allow_pickle=False)
        assert data["samples"].shape == (7, 7, 101)
        assert np.isfinite(data["samples"]).all()
        assert data["samples"].min() >= 0
    latest = pd.read_parquet(bundle / "latest_leaf_forecasts.parquet")
    assert len(latest) == 2 * 7 and latest.model.eq("mint_shrink").all()
    scored = pd.read_parquet(bundle / "predictions.parquet")
    naive = scored.query("model == 'base' and unique_id in ['A','B']").sort_values(
        ["fold_id", "unique_id", "ds"]
    )
    bottom_up = scored.query("model == 'bottom_up' and unique_id in ['A','B']").sort_values(
        ["fold_id", "unique_id", "ds"]
    )
    np.testing.assert_array_equal(naive.point, bottom_up.point)
    # Pre-origin fitted errors and learned projections cannot read holdout targets.
    catalog = (
        store.history[["unique_id", "store_id", "item_id"]]
        .drop_duplicates()
        .merge(store.items, on="item_id")
    )
    hierarchy = Hierarchy.from_catalog(catalog)
    base = pd.read_parquet(forecast / "predictions.parquet").query("fold_id == 4")
    origin = date(2020, 11, 29)
    before = prepare_reconciliation_data(
        store, hierarchy, origin, base, upstream, forecast / "holdout"
    )
    history, calendar, prices, items = [frame.copy() for frame in model_inputs]
    history.loc[history.ds > pd.Timestamp(origin), "y"] += 10000
    changed = FeatureStore.from_frames(history, calendar, prices, items, provenance=provenance)
    after = prepare_reconciliation_data(
        changed, hierarchy, origin, base, upstream, forecast / "holdout"
    )
    np.testing.assert_allclose(before.actual_insample, after.actual_insample)
    np.testing.assert_allclose(before.fitted_insample, after.fitted_insample)
    first = fit_reconcilers(
        hierarchy, before.base_point, before.actual_insample, before.fitted_insample
    )
    second = fit_reconcilers(
        hierarchy, after.base_point, after.actual_insample, after.fitted_insample
    )
    np.testing.assert_allclose(first.covariance, second.covariance)
    for method in first.projections:
        np.testing.assert_allclose(first.projections[method], second.projections[method])
    assert set(nodes.level) == {"total", "store", "store_category", "store_item"}
