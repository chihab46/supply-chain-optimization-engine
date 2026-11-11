from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from supply_chain_opt_engine.baseline import (
    MODELS,
    BaselineConfig,
    forecast_fold,
    run_backtest,
    save_backtest,
)
from supply_chain_opt_engine.evaluation import seasonal_folds, validate_history
from supply_chain_opt_engine.m5 import checksum
from supply_chain_opt_engine.metrics import QUANTILES, quantile_column


def folds(horizon):
    dates = [date(2020, 2, 29), date(2020, 5, 31), date(2020, 8, 16), date(2020, 11, 29)]
    return seasonal_folds(dates, horizon, dates[0])


def test_real_models_generate_nonnegative_ordered_quantiles_without_future_leakage(history_panel):
    history = validate_history(history_panel)
    config = BaselineConfig(horizon=7)
    fold = folds(7)[0]
    first, scales, _ = forecast_fold(history, fold, config)
    altered = history.copy()
    altered.loc[altered.ds > pd.Timestamp(fold.origin), "y"] += 10000
    second, changed_scales, _ = forecast_fold(altered, fold, config)
    columns = ["point", *[quantile_column(q) for q in QUANTILES]]
    np.testing.assert_allclose(first[columns], second[columns])
    pd.testing.assert_frame_equal(scales, changed_scales)
    assert len(first) == 2 * 7 * len(MODELS)
    assert first[columns].ge(0).all().all()
    assert (np.diff(first[[quantile_column(q) for q in QUANTILES]], axis=1) >= -1e-10).all()
    naive = first[(first.model == "SeasonalNaive") & (first.unique_id == "A")].sort_values("ds")
    recent = history[(history.unique_id == "A") & (history.ds <= pd.Timestamp(fold.origin))].tail(7)
    np.testing.assert_allclose(naive.point, recent.y, rtol=1e-6)


def test_four_fold_backtest_exports_inspectable_scores_and_holdout_summary(history_panel, tmp_path):
    config = BaselineConfig(horizon=7)
    schedule = folds(7)
    tables = run_backtest(history_panel, schedule, config)
    assert len(tables["fold_metrics"]) == 4 * len(MODELS)
    assert len(tables["summary_metrics"]) == 3 * len(MODELS)
    assert set(tables["summary_metrics"].scope) == {"all_folds", "tuning_folds", "holdout"}
    assert tables["series_metrics"].groupby("fold_id").role.nunique().eq(1).all()
    assert tables["series_metrics"].query("fold_id == 4").role.eq("holdout").all()
    bundle = save_backtest(
        tables, tmp_path / "baselines", schedule, config, {"source_type": "test"}
    )
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["status"] == "backtested"
    assert manifest["prediction_rows"] == 2 * 7 * 4 * len(MODELS)
    for name, digest in manifest["artifact_sha256"].items():
        assert checksum(bundle / name) == digest
    assert len(pd.read_parquet(bundle / "predictions.parquet")) == manifest["prediction_rows"]
    assert not list(bundle.parent.glob(".baseline-*"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("conformal_windows", 1),
        ("horizon", 0),
        ("n_jobs", True),
        ("tsb_alpha_p", 0.0),
    ],
)
def test_invalid_baseline_configuration_fails_before_training(field, value):
    with pytest.raises(ValueError):
        BaselineConfig(**{field: value})
