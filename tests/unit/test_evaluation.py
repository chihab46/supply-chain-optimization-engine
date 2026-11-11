from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from supply_chain_opt_engine.evaluation import (
    seasonal_folds,
    split_fold,
    training_scales,
    validate_history,
)


def cutoffs():
    return [date(2020, 2, 29), date(2020, 5, 31), date(2020, 8, 16), date(2020, 11, 29)]


def test_four_seasonal_folds_are_chronological_and_nonoverlapping(history_panel):
    history = validate_history(history_panel)
    folds = seasonal_folds(cutoffs(), 28, date(2020, 2, 29))
    assert [fold.role for fold in folds] == ["tuning"] * 3 + ["holdout"]
    assert [fold.season for fold in folds] == [
        "spring",
        "summer",
        "back_to_school",
        "winter_holidays",
    ]
    for fold in folds:
        train, actual = split_fold(history, fold, min_train_days=100)
        assert train.ds.max().date() == fold.origin
        assert actual.ds.min().date() == fold.origin + timedelta(days=1)
        assert actual.ds.max().date() == fold.origin + timedelta(days=28)
        assert actual.groupby("unique_id").size().eq(28).all()


def test_folds_reject_future_selection_overlap_and_wrong_seasons():
    with pytest.raises(ValueError, match="selection used data later"):
        seasonal_folds(cutoffs(), 28, date(2020, 3, 1))
    overlapping = [date(2020, 2, 29), date(2020, 3, 10), *cutoffs()[2:]]
    with pytest.raises(ValueError, match="overlap"):
        seasonal_folds(overlapping, 28, date(2020, 2, 29))
    wrong_season = [date(2020, 1, 1), *cutoffs()[1:]]
    with pytest.raises(ValueError, match="spring"):
        seasonal_folds(wrong_season, 28, date(2019, 1, 1))
    with pytest.raises(ValueError, match="four distinct"):
        seasonal_folds(cutoffs()[::-1], 28, date(2019, 1, 1))


def test_scales_use_only_training_seasonal_differences(history_panel):
    history = validate_history(history_panel)
    fold = seasonal_folds(cutoffs(), 28, date(2020, 2, 29))[0]
    train, actual = split_fold(history, fold, 100)
    scales = training_scales(train, 7).set_index("unique_id")
    assert scales.loc["A", "rmsse_scale"] == pytest.approx(0.01)
    assert scales.loc["B", "rmsse_scale"] == 0
    actual["y"] += 100000
    # Modifying validation targets cannot change the scale fitted to training.
    changed = pd.concat([train, actual], ignore_index=True)
    train_again, _ = split_fold(changed, fold, 100)
    pd.testing.assert_frame_equal(training_scales(train, 7), training_scales(train_again, 7))


@pytest.mark.parametrize("problem", ["duplicate", "gap", "negative", "nan", "flag"])
def test_history_rejects_broken_daily_series(history_panel, problem):
    if problem == "duplicate":
        history_panel = pd.concat([history_panel, history_panel.iloc[[0]]])
    elif problem == "gap":
        history_panel = history_panel.drop(index=4)
    elif problem == "negative":
        history_panel.loc[0, "y"] = -1
    elif problem == "nan":
        history_panel.loc[0, "y"] = np.nan
    else:
        history_panel["is_available"] = "false"
    with pytest.raises(ValueError):
        validate_history(history_panel)


def test_missing_holdout_days_fail_before_training(history_panel):
    folds = seasonal_folds(cutoffs(), 28, date(2020, 2, 29))
    truncated = validate_history(history_panel[history_panel.ds < "2020-12-10"])
    with pytest.raises(ValueError, match="validation horizon"):
        split_fold(truncated, folds[3], 100)
