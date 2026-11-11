"""Nixtla reconciliation projections and joint rank-block sample paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hierarchicalforecast.methods import BottomUp, MinTrace, TopDown
from scipy.sparse import csr_matrix

from supply_chain_opt_engine.hierarchy import Hierarchy
from supply_chain_opt_engine.metrics import QUANTILES

METHODS = ("bottom_up", "top_down", "mint_shrink")


@dataclass
class ReconciliationFit:
    projections: dict[str, np.ndarray]
    covariance: np.ndarray
    top_down_uniform_fallback: bool
    mint_ols_fallback: bool


def fit_reconcilers(
    hierarchy: Hierarchy,
    base_point: np.ndarray,
    actual: np.ndarray,
    fitted: np.ndarray,
    ridge: float = 2e-8,
) -> ReconciliationFit:
    if (
        actual.shape != fitted.shape
        or actual.shape[0] != len(hierarchy.nodes)
        or actual.shape[1] < 2
        or not np.isfinite(actual).all()
        or not np.isfinite(fitted).all()
    ):
        raise ValueError(
            "Covariance requires finite, time-aligned training-only residuals at all nodes."
        )
    models = {
        "bottom_up": BottomUp(),
        "top_down": TopDown(method="proportion_averages"),
        "mint_shrink": MinTrace(method="mint_shrink", mint_shr_ridge=ridge, num_threads=1),
    }
    projections = {}
    covariance = None
    fallback = bool(actual[0].sum() == 0)
    mint_fallback = bool((np.abs((actual - fitted).sum(axis=1)) < 1e-4).mean() > 0.98)
    for name, model in models.items():
        if name == "top_down" and fallback:
            P = np.zeros((hierarchy.n_bottom, len(hierarchy.nodes)))
            P[:, 0] = 1 / hierarchy.n_bottom
        else:
            if name == "mint_shrink" and mint_fallback:
                # Nixtla rejects near-zero residual sums. Expose an explicit
                # OLS fallback for that degenerate case rather than inventing errors.
                model = MinTrace(method="ols", num_threads=1)
            model.fit(
                S=hierarchy.S,
                y_hat=base_point,
                y_insample=actual,
                y_hat_insample=fitted,
                tags=hierarchy.tags,
            )
            P = np.asarray(model.P, dtype=float)
            if name == "mint_shrink":
                covariance = np.asarray(model.W, dtype=float)
        if P.shape != (hierarchy.n_bottom, len(hierarchy.nodes)) or not np.isfinite(P).all():
            raise ValueError(f"{name} produced an invalid reconciliation projection.")
        projections[name] = P
    return ReconciliationFit(projections, covariance, fallback, mint_fallback)


def reconcile_values(
    hierarchy: Hierarchy, P: np.ndarray, base: np.ndarray
) -> tuple[np.ndarray, int]:
    """Apply the learned linear projection, clip bottom negatives, then reaggregate."""
    shape = base.shape
    if shape[0] != len(hierarchy.nodes) or not np.isfinite(base).all():
        raise ValueError("Base forecasts must be finite and ordered by hierarchy node.")
    flat = np.asarray(base, dtype=float).reshape(len(hierarchy.nodes), -1)
    n_agg = len(hierarchy.nodes) - hierarchy.n_bottom
    A = P[:, :n_agg]
    if np.allclose(
        P[:, n_agg:], np.eye(hierarchy.n_bottom) - A @ hierarchy.S[:n_agg], rtol=1e-10, atol=1e-10
    ):
        bottom = flat[n_agg:] + A @ (flat[:n_agg] - hierarchy.S[:n_agg] @ flat[n_agg:])
    elif np.count_nonzero(P) < P.size / 10:
        bottom = csr_matrix(P) @ flat
    else:
        bottom = P @ flat
    clipped = int((bottom < 0).sum())
    reconciled = hierarchy.aggregate(
        np.maximum(bottom, 0).reshape((hierarchy.n_bottom, *shape[1:]))
    )
    return reconciled, clipped


def rank_block_uniforms(
    residuals: np.ndarray, horizon: int, num_samples: int, seed: int
) -> np.ndarray:
    """Resample aligned residual-rank blocks, with randomized ranks within exact ties."""
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2 or not np.isfinite(residuals).all() or residuals.shape[1] < horizon:
        raise ValueError(
            "Residual history must contain a finite aligned block at least as long as the horizon."
        )
    if num_samples < 2 or horizon < 1:
        raise ValueError("Require positive horizon and at least two joint sample paths.")
    rng = np.random.default_rng(seed)
    n = residuals.shape[1]
    order = np.lexsort((rng.random(residuals.shape), residuals), axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(n), order.shape), axis=1)
    uniforms = (ranks + 0.5) / n
    starts = rng.integers(0, n, size=num_samples)
    indices = (np.arange(horizon)[:, None] + starts[None, :]) % n
    return uniforms[:, indices]


def base_sample_paths(quantiles: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Invert piecewise-linear marginal CDFs without adding marginal quantiles."""
    if quantiles.shape[:2] != uniforms.shape[:2] or quantiles.shape[2] != len(QUANTILES):
        raise ValueError("Quantile and latent-uniform node/horizon dimensions disagree.")
    if not np.isfinite(uniforms).all() or (uniforms < 0).any() or (uniforms > 1).any():
        raise ValueError("Latent ranks must be finite probabilities in [0, 1].")
    if (
        not np.isfinite(quantiles).all()
        or (quantiles < 0).any()
        or (np.diff(quantiles, axis=2) < -1e-10).any()
    ):
        raise ValueError("Base quantiles must be finite, nonnegative and ordered.")
    probabilities = np.array([0, *QUANTILES, 1])
    n_samples = uniforms.shape[2]
    if n_samples < 101 or (n_samples - 1) % 100 != 0:
        raise ValueError("Use 100k+1 paths to represent the declared marginal quantiles exactly.")
    tail = quantiles[:, :, -1] + (quantiles[:, :, -1] - quantiles[:, :, -2])
    anchors = np.concatenate(
        [np.zeros((*quantiles.shape[:2], 1)), quantiles, tail[:, :, None]], axis=2
    )
    result = np.empty_like(uniforms, dtype=float)
    grid = np.linspace(0, 1, n_samples)
    for probability in QUANTILES:
        grid[round(probability * (n_samples - 1))] = probability
    for node in range(quantiles.shape[0]):
        for day in range(quantiles.shape[1]):
            # Preserve the bootstrap copula ranks while making marginal
            # probability grids identical. Quantile anchors, including zero
            # atoms, are represented exactly with 100k+1 paths.
            order = np.argsort(uniforms[node, day], kind="stable")
            ordered = np.interp(grid, probabilities, anchors[node, day])
            result[node, day, order] = ordered
    return result
