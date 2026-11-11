from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from supply_chain_opt_engine.features import validate_panel


@pytest.fixture
def panel():
    return pd.DataFrame(
        {
            "unique_id": ["CA_1/FOODS_1_001"] * 3 + ["CA_2/FOODS_1_001"] * 3,
            "forecast_origin": [date(2020, 1, 29)] * 6,
            "ds": list(pd.date_range("2020-01-30", periods=3)) * 2,
            "horizon": [1, 2, 3] * 2,
            "price_momentum": [None] * 6,
            "rolling_mean_7": [3.0] * 6,
        }
    )


def test_valid_panel_allows_explicit_missing_features(panel):
    validate_panel(panel, date(2020, 1, 29), 3)


def test_panel_rejects_target_labels(panel):
    panel["y"] = 5
    with pytest.raises(ValueError, match="realized target"):
        validate_panel(panel, date(2020, 1, 29), 3)


def test_panel_rejects_duplicate_or_null_keys(panel):
    with pytest.raises(ValueError, match="duplicate"):
        validate_panel(pd.concat([panel, panel.iloc[[0]]]), date(2020, 1, 29), 3)
    panel.loc[0, "unique_id"] = None
    with pytest.raises(ValueError, match="Null"):
        validate_panel(panel, date(2020, 1, 29), 3)


def test_panel_rejects_missing_horizon_or_wrong_target_date(panel):
    with pytest.raises(ValueError, match="complete requested"):
        validate_panel(panel.iloc[:-1], date(2020, 1, 29), 3)
    panel.loc[0, "ds"] = pd.Timestamp("2020-02-15")
    with pytest.raises(ValueError, match="Target dates"):
        validate_panel(panel, date(2020, 1, 29), 3)


def test_panel_rejects_mixed_origins_and_infinite_values(panel):
    panel.loc[0, "forecast_origin"] = date(2020, 1, 28)
    with pytest.raises(ValueError, match="inconsistent forecast origins"):
        validate_panel(panel, date(2020, 1, 29), 3)
    panel["forecast_origin"] = date(2020, 1, 29)
    panel.loc[0, "rolling_mean_7"] = np.inf
    with pytest.raises(ValueError, match="Infinite feature"):
        validate_panel(panel, date(2020, 1, 29), 3)
