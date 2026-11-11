from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from supply_chain_opt_engine.metrics import (
    QUANTILES,
    aggregate_scores,
    pinball_loss,
    quantile_column,
    score_predictions,
)


def example_scores():
    frame = pd.DataFrame(
        {
            "fold_id": [1] * 4,
            "unique_id": ["A", "A", "B", "B"],
            "model": ["Example"] * 4,
            "ds": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")] * 2,
            "y": [10.0, 20.0, 0.0, 0.0],
            "point": [12.0, 18.0, 1.0, 1.0],
        }
    )
    for q in QUANTILES:
        frame[quantile_column(q)] = frame["point"]
    frame["q05"] = 0.0
    frame["q95"] = 20.0
    scales = pd.DataFrame(
        {
            "fold_id": [1, 1],
            "unique_id": ["A", "B"],
            "rmsse_scale": [4.0, 0.0],
            "training_days": [100, 100],
        }
    )
    return frame, scales


def test_pinball_has_correct_asymmetric_penalties():
    assert pinball_loss(np.array([10]), np.array([8]), 0.9) == pytest.approx(1.8)
    assert pinball_loss(np.array([0]), np.array([2]), 0.9) == pytest.approx(0.2)
    assert pinball_loss(np.array([10, 0]), np.array([8, 2]), 0.9) == pytest.approx(1.0)
    assert pinball_loss(np.array([10, 0]), np.array([8, 2]), 0.5) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="strictly"):
        pinball_loss(np.array([1]), np.array([1]), 1.0)


def test_zero_denominators_remain_undefined_and_are_counted():
    frame, scales = example_scores()
    scores = score_predictions(frame, scales).set_index("unique_id")
    assert scores.loc["A", "wmape"] == pytest.approx(4 / 30)
    assert scores.loc["A", "rmsse"] == 1.0
    assert np.isnan(scores.loc["B", "wmape"])
    assert np.isnan(scores.loc["B", "rmsse"])
    assert scores.loc["A", "pinball_q50"] == 1.0
    assert scores["coverage_90"].eq(1).all()  # Includes boundaries at zero and twenty.
    summary = aggregate_scores(scores.reset_index(), ["model"]).iloc[0]
    # Errors on the zero-demand series still contribute to pooled WMAPE.
    assert summary.wmape == pytest.approx(6 / 30)
    assert summary.rmsse == 1.0
    assert summary.undefined_wmape_series == summary.undefined_rmsse_series == 1


def test_macro_rmsse_is_equal_series_weighted_and_wmape_is_pooled():
    frame, scales = example_scores()
    scales.loc[1, "rmsse_scale"] = 1.0
    frame.loc[frame.unique_id == "B", "point"] = 3.0
    scores = score_predictions(frame, scales)
    summary = aggregate_scores(scores, ["model"]).iloc[0]
    assert summary.rmsse == 2.0  # Mean of series RMSSE 1 and 3.
    assert summary.wmape == pytest.approx(10 / 30)


def test_metrics_reject_crossing_quantiles_duplicates_and_missing_scales():
    frame, scales = example_scores()
    frame.loc[0, "q05"] = 100
    with pytest.raises(ValueError, match="quantiles cross"):
        score_predictions(frame, scales)
    frame.loc[0, "q05"] = 0
    with pytest.raises(ValueError, match="Duplicate"):
        score_predictions(pd.concat([frame, frame.iloc[[0]]]), scales)
    with pytest.raises(ValueError, match="scales are missing"):
        score_predictions(frame, scales.iloc[:1])
