"""TreeSHAP diagnostics for the raw P50 LightGBM model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from supply_chain_opt_engine.learning import DirectDataset


@dataclass
class ShapDiagnostics:
    rows: pd.DataFrame
    importance: pd.DataFrame


def explain_median(model, data: DirectDataset, max_rows: int, seed: int) -> ShapDiagnostics:
    # Lazy import avoids initializing plotting/font caches when fitting models.
    import shap

    # Prefer one day-one row per series, so the dashboard has SKU-level drivers.
    candidates = data.keys.index[data.keys["horizon"].eq(1)].to_numpy()
    if len(candidates) > max_rows:
        candidates = np.sort(
            np.random.default_rng(seed).choice(candidates, max_rows, replace=False)
        )
    X = data.X.iloc[candidates]
    explainer = shap.TreeExplainer(
        model, feature_perturbation="tree_path_dependent", model_output="raw"
    )
    contributions = np.asarray(explainer.shap_values(X), dtype=float)
    baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    prediction = model.predict(X)
    if contributions.shape != X.shape or not np.allclose(
        baseline + contributions.sum(axis=1),
        prediction,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError("SHAP contributions do not reconstruct the raw median forecast.")
    keys = data.keys.iloc[candidates].reset_index(drop=True)
    rows = []
    for index, feature in enumerate(X.columns):
        values = X[feature].astype(object).where(X[feature].notna(), None)
        part = keys.copy()
        part["feature"] = feature
        part["feature_value"] = values.reset_index(drop=True).map(
            lambda value: None if value is None else str(value)
        )
        part["shap_value"] = contributions[:, index]
        part["expected_value"] = baseline
        part["raw_q50"] = prediction
        part["reported_point"] = np.maximum(prediction, 0)
        rows.append(part)
    importance = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "mean_absolute_shap": np.abs(contributions).mean(axis=0),
                "sample_rows": len(X),
            }
        )
        .sort_values("mean_absolute_shap", ascending=False)
        .reset_index(drop=True)
    )
    return ShapDiagnostics(pd.concat(rows, ignore_index=True), importance)
