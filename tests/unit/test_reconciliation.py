from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from supply_chain_opt_engine.hierarchy import Hierarchy
from supply_chain_opt_engine.reconcilers import (
    base_sample_paths,
    fit_reconcilers,
    rank_block_uniforms,
    reconcile_values,
)


@pytest.fixture
def hierarchy():
    return Hierarchy.from_catalog(
        pd.DataFrame(
            {
                "unique_id": ["a", "b", "c", "d"],
                "store_id": ["CA_1", "CA_1", "CA_2", "CA_2"],
                "cat_id": ["FOODS", "HOUSEHOLD", "FOODS", "HOUSEHOLD"],
                "item_id": ["i1", "i2", "i1", "i2"],
            }
        )
    )


def test_hierarchy_has_stable_indices_and_exact_summing_matrix(hierarchy):
    assert len(hierarchy.nodes) == 11
    assert hierarchy.nodes.level.value_counts().to_dict() == {
        "store_category": 4,
        "store_item": 4,
        "store": 2,
        "total": 1,
    }
    np.testing.assert_array_equal(hierarchy.S[-4:], np.eye(4))
    values = hierarchy.aggregate(np.arange(1, 5, dtype=float))
    assert values[0] == 10
    np.testing.assert_array_equal(values[1:3], [3, 7])
    hierarchy.assert_coherent(values)
    shuffled = hierarchy.nodes.query("level == 'store_item'").sample(frac=1, random_state=4)
    other = Hierarchy.from_catalog(shuffled)
    pd.testing.assert_frame_equal(other.nodes, hierarchy.nodes)
    np.testing.assert_array_equal(other.S, hierarchy.S)


def test_invalid_catalog_and_incoherent_values_are_rejected(hierarchy):
    bad = hierarchy.nodes.query("level == 'store_item'").copy()
    bad.loc[bad.index[0], "cat_id"] = None
    with pytest.raises(ValueError, match="complete"):
        Hierarchy.from_catalog(bad)
    values = hierarchy.aggregate(np.ones((4, 7)))
    values[0, 0] += 1
    with pytest.raises(ValueError, match="coherently"):
        hierarchy.assert_coherent(values)


def test_nixtla_projections_and_nonnegative_reaggregation(hierarchy):
    rng = np.random.default_rng(42)
    actual = hierarchy.aggregate(rng.poisson(3, (4, 90)))
    fitted = actual + rng.normal(0, 0.5, actual.shape)
    base = rng.uniform(0, 20, (11, 7))
    fit = fit_reconcilers(hierarchy, base, actual, fitted)
    for name, P in fit.projections.items():
        result, _ = reconcile_values(hierarchy, P, base)
        expected = hierarchy.aggregate(np.maximum(P @ base, 0))
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
        assert result.min() >= 0
        hierarchy.assert_coherent(result)
        if name == "bottom_up":
            np.testing.assert_array_equal(result[-4:], base[-4:])
    assert fit.covariance.shape == (11, 11)
    assert np.isfinite(fit.covariance).all()


def test_degenerate_history_fallbacks_are_explicit_and_coherent(hierarchy):
    actual = np.zeros((11, 60))
    base = np.ones((11, 7))
    fit = fit_reconcilers(hierarchy, base, actual, actual)
    assert fit.top_down_uniform_fallback
    assert fit.mint_ols_fallback
    for P in fit.projections.values():
        result, _ = reconcile_values(hierarchy, P, base)
        hierarchy.assert_coherent(result)


def test_rank_block_sampler_is_seeded_circular_and_handles_ties():
    residual = np.tile(np.arange(70, dtype=float), (3, 1))
    u = rank_block_uniforms(residual, 7, 100, 42)
    np.testing.assert_array_equal(u, rank_block_uniforms(residual, 7, 100, 42))
    assert not np.array_equal(u, rank_block_uniforms(residual, 7, 100, 43))
    assert u.min() > 0 and u.max() < 1
    np.testing.assert_allclose((np.diff(u, axis=1) + 1) % 1, 1 / 70, atol=1e-12)
    tied = rank_block_uniforms(np.zeros((3, 70)), 7, 100, 42)
    assert tied.std() > 0
    assert not np.array_equal(tied[0], tied[1])


def test_coherent_paths_do_not_imply_additive_marginal_medians(hierarchy):
    bottom = np.zeros((4, 1, 3))
    bottom[0, 0] = [0, 0, 10]
    bottom[1, 0] = [0, 10, 0]
    coherent = hierarchy.aggregate(bottom)
    hierarchy.assert_coherent(coherent)
    medians = np.median(coherent, axis=2)
    assert medians[0, 0] == 10
    assert medians[-4:, 0].sum() == 0
    assert hierarchy.coherence_error(medians) == 10


def test_inverse_cdf_samples_reconcile_jointly(hierarchy):
    rng = np.random.default_rng(3)
    actual = hierarchy.aggregate(rng.poisson(3, (4, 90)))
    fit = fit_reconcilers(
        hierarchy, actual[:, :7] + 1, actual, actual + rng.normal(0, 1, actual.shape)
    )
    q = np.broadcast_to(np.array([0, 1, 3, 5, 7, 9]), (11, 7, 6))
    paths = base_sample_paths(
        q, rank_block_uniforms(actual - actual.mean(axis=1, keepdims=True), 7, 101, 42)
    )
    for P in fit.projections.values():
        samples, _ = reconcile_values(hierarchy, P, paths)
        hierarchy.assert_coherent(samples)
        assert samples.min() >= 0
        quantiles = np.quantile(samples, [0.05, 0.1, 0.5, 0.8, 0.9, 0.95], axis=2)
        assert (np.diff(quantiles, axis=0) >= -1e-12).all()


def test_rank_stratification_preserves_quantiles_including_zero_atoms():
    q = np.broadcast_to(np.array([0, 1, 3, 5, 7, 9]), (3, 7, 6)).copy()
    q[1, :, :3] = 0
    rng = np.random.default_rng(42)
    u = rank_block_uniforms(rng.normal(size=(3, 90)), 7, 501, 42)
    samples = base_sample_paths(q, u)
    actual = np.moveaxis(np.quantile(samples, [0.05, 0.1, 0.5, 0.8, 0.9, 0.95], axis=2), 0, -1)
    np.testing.assert_allclose(actual, q, rtol=0, atol=1e-12)
