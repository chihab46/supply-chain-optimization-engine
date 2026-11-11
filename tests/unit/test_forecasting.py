from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

from supply_chain_opt_engine.evaluation import Fold
from supply_chain_opt_engine.feature_store import FeatureEncoder, FeatureStore
from supply_chain_opt_engine.forecasting import run_forecasting
from supply_chain_opt_engine.learning import (
    ForecastConfig,
    predict_saved,
    prediction_dataset,
    repair_quantiles,
    training_dataset,
)
from supply_chain_opt_engine.m5 import checksum
from supply_chain_opt_engine.tuning import FoldData, tune_parameters


def make_store(inputs):
    return FeatureStore.from_frames(
        *inputs, provenance={"source_type": "test", "selection_cutoff": "2019-01-01"}
    )


def small_config():
    return ForecastConfig(
        horizon=7,
        origin_stride_days=7,
        training_window_days=28,
        n_estimators=6,
        optuna_trials=1,
        num_threads=1,
        shap_rows=2,
    )


def test_training_features_and_labels_cannot_read_past_cutoff(model_inputs):
    original = make_store(model_inputs)
    cutoff = date(2020, 2, 29)
    config = small_config()
    encoder = FeatureEncoder.from_store(original)
    before = training_dataset(original, encoder, cutoff, config)
    h, c, p, e = [frame.copy() for frame in model_inputs]
    h.loc[h.ds > pd.Timestamp(cutoff), "y"] += 10000
    week = c.loc[c.date.eq(pd.Timestamp(cutoff)), "wm_yr_wk"].item()
    p.loc[p.wm_yr_wk > week, "sell_price"] += 10000
    changed = make_store((h, c, p, e))
    after = training_dataset(changed, encoder, cutoff, config)
    assert_frame_equal(before.X, after.X)
    assert_frame_equal(before.keys, after.keys)
    np.testing.assert_array_equal(before.y, after.y)
    assert before.keys.ds.max().date() <= cutoff
    # Prediction features are also invariant, while held-out target labels may differ.
    assert_frame_equal(
        prediction_dataset(original, encoder, cutoff, config).X,
        prediction_dataset(changed, encoder, cutoff, config).X,
    )


def test_quantile_repair_preserves_p50_and_orders_bounds():
    raw = pd.DataFrame(
        {
            "q05": [4, -2],
            "q10": [2, -1],
            "q50": [3, -0.5],
            "q80": [2, -3],
            "q90": [8, 2],
            "q95": [6, 1],
        }
    )
    repaired = repair_quantiles(raw)
    np.testing.assert_array_equal(repaired.q50, [3, 0])
    assert (np.diff(repaired.to_numpy(), axis=1) >= 0).all()
    assert repaired.ge(0).all().all()
    assert_frame_equal(repair_quantiles(repaired), repaired)


def test_optuna_rejects_holdout_before_creating_a_study(model_inputs, tmp_path):
    store = make_store(model_inputs)
    encoder = FeatureEncoder.from_store(store)
    config = small_config()
    dataset = prediction_dataset(store, encoder, date(2020, 2, 29), config, with_labels=True)
    data = [
        FoldData(Fold(4, date(2020, 11, 29), 7, "winter_holidays", "holdout"), dataset, dataset)
    ]
    with pytest.raises(ValueError, match="never the holdout"):
        tune_parameters(data, config, tmp_path / "rejected.sqlite3")
    assert not (tmp_path / "rejected.sqlite3").exists()


def test_full_global_workflow_saves_models_study_shap_and_frozen_scores(model_inputs, tmp_path):
    store = make_store(model_inputs)
    document = yaml.safe_load(Path("configs/default.yaml").read_text())
    document["seed"] = 42
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
    bundle = run_forecasting(store, document, tmp_path / "models")
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["prediction_rows"] == 2 * 7 * 4
    assert manifest["latest_rows"] == 2 * 7
    for name, digest in manifest["artifact_sha256"].items():
        assert checksum(bundle / name) == digest
    folds = pd.read_csv(bundle / "fold_metrics.csv")
    assert folds.query("fold_id == 4").role.eq("holdout").all()
    trials = pd.read_csv(bundle / "optuna_trials.csv")
    assert "user_attrs_fold_4_pinball_q50" not in trials
    assert all(
        manifest["training_blocks"][index]["max_label_date"] <= fold["origin"]
        for index, fold in enumerate(manifest["folds"])
    )
    latest = pd.read_parquet(bundle / "latest_forecasts.parquet")
    panel = store.panel(date.fromisoformat(manifest["latest_origin"]), 7)
    loaded = predict_saved(bundle / "latest", panel)
    np.testing.assert_allclose(latest[loaded.columns], loaded, rtol=1e-10, atol=1e-10)
    shap = pd.read_parquet(bundle / "holdout_shap.parquet")
    reconstructed = shap.groupby(["unique_id", "ds"]).agg(
        total=("shap_value", "sum"), base=("expected_value", "first"), raw=("raw_q50", "first")
    )
    np.testing.assert_allclose(
        reconstructed.total + reconstructed.base, reconstructed.raw, rtol=1e-6
    )
    assert not list(bundle.parent.glob(".lightgbm-*"))


@pytest.mark.parametrize(
    "options", [{"horizon": 29}, {"origin_stride_days": 7}, {"optuna_trials": 0}]
)
def test_invalid_global_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        ForecastConfig(**options)
