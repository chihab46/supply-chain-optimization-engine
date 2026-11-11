# Hierarchical reconciliation

Step 5 builds the retail hierarchy, generates independent aggregate forecasts,
fits Bottom-Up, Top-Down and MinT-shrink reconciliation, and evaluates every
level. It produces coherent point forecasts and coherent joint paths. Marginal
quantiles are reported separately and are not assumed to add across children.

## Hierarchy and base models

The selected dataset has 613 nodes: Total, 3 stores, 9 store-category groups and
600 store-item leaves. Department remains item metadata rather than a fifth
hierarchy level. Leaf IDs match the global forecasting artifacts. Aggregate IDs
have explicit `Store/` and `StoreCategory/` prefixes. The summing matrix has shape
613 by 600, with an identity block for the leaves. Node indices and parent links
are saved alongside the matrix and level tags.

Historical demand is aggregated exactly from the selected leaves. Aggregate
forecasts are independent AutoETS models with weekly seasonality, fitted at each
origin using only earlier observations. Leaf point and quantile forecasts are
the frozen global LightGBM outputs from Step 4. Adding up leaf forecasts to create
every base aggregate would make reconciliation a trivial identity, so the upper
nodes receive independent forecasts instead.

These base point estimates mix LightGBM medians and AutoETS centers. A median is
not generally additive, and the classical unbiased-mean assumptions behind
MinT's optimality argument do not automatically hold for this mixture. Accuracy
changes are measured rather than assumed to improve.

## Training-only fitted errors

The reconciliation window matches the global model's direct training-label dates,
normally 364 aligned days. Leaf fitted predictions use the saved holdout or
production P50 model. For folds 1 to 3, the runner deterministically refits only
the median model using the already frozen Step 4 parameters. Its future median
predictions must reproduce the saved upstream forecasts before fitted errors
are accepted. No new Optuna search occurs.

Aggregate AutoETS fitted values are aligned to those same training dates. Every
residual date must be on or before the corresponding forecast origin. These are
in-sample fitted errors, not an independent calibration holdout. Their use and
the upstream tuning-fold selection bias remain visible in the run manifest.
Changing post-origin target demand cannot affect residuals, covariance or shares.

## Reconciliation methods

Nixtla's `hierarchicalforecast` methods fit the reconciliation matrices:

- Bottom-Up keeps leaf forecasts and aggregates them through the summing matrix.
- Top-Down allocates the independent total forecast using each leaf's proportion
  of training-average demand over the aligned window.
- MinT uses `MinTrace(method='mint_shrink')` and the aligned fitted-error matrix
  to estimate a shrunk covariance and reconciliation projection.

The learned projection is applied to point forecasts and to every joint path.
Negative projected leaf values are clipped to zero, then all aggregates are
recomputed. This preserves nonnegative demand and coherence but is not the
constrained quadratic-programming optimum. Clipping counts are saved by phase
and method. The raw shrinkage covariance and all projection matrices are saved.

If total training demand is zero, Top-Down uses explicitly recorded uniform
shares. If Nixtla rejects near-zero residual sums, the MinT branch uses an
explicitly recorded OLS fallback. Such fallbacks must not be described as an
ordinary estimated MinT-shrink result. Native ETS intervals that are numerically
undefined use centered empirical training-error quantiles for those cells only;
fallback counts are recorded. None of these rules uses future target values.

## Joint paths and marginal quantiles

Base marginal CDFs interpolate the six declared quantiles. The lower endpoint is
P00=0; the upper endpoint extrapolates P100 as `P95 + (P95 - P90)`. This is a
bounded-tail modeling assumption, not an inferred distribution beyond P95.

Aligned pre-origin residuals define an empirical copula. Exact residual ties get
seeded randomized ranks. Circular historical blocks preserve local temporal and
cross-node rank dependence without reading future observations. Scenario ranks
are then stratified onto a common uniform probability grid. This retains the
empirical copula ordering while representing the supplied marginal quantiles.

The default is 501 paths. Counts must be `100k + 1`, such as 101, 201 or 501, so
P05, P10, P50, P80, P90 and P95 lie on grid points. Known quantile anchors are
snapped to their exact probabilities. This avoids a finite-sample artifact where
a zero lower quantile becomes slightly positive and stops covering observed
zero sales. A property test requires Bottom-Up leaf quantiles to reproduce their
unreconciled marginals within numerical tolerance, without using actual targets.

All paths are projected, repaired at the leaves, and reaggregated. The runner
then calculates marginal quantiles at each node from the coherent samples.
Quantiles are ordered and nonnegative. Parent marginal medians and upper
quantiles need not equal sums of child marginal quantiles, even though every
underlying sample is coherent. A test demonstrates this distinction explicitly.

The exported `point` is a coherent projection of the base centers. It is not
necessarily the median of the reconciled distribution. `q50` is the marginal
sample median and is not asserted to be coherent. Point WMAPE and RMSSE use
`point`; pinball losses and central P05-P95 coverage use the marginal quantiles.
The optimizer's leaf P90 input will be a quantile-demand planning scenario,
not a guarantee of a 90% realized fill rate.

## Evaluation and execution

The same four frozen seasonal windows are used. `base`, `bottom_up`, `top_down`
and `mint_shrink` scores are saved at Total, Store, Store-Category and Store-Item
levels. The original full pre-origin seasonal RMSSE scales are retained. Reports
include per-fold, tuning-only, holdout and all-fold summaries. Reconciliation
methods and the primary output method are declared before scoring; the runner
does not select a winner from holdout metrics.

```bash
uv run python -m supply_chain_opt_engine.reconciliation
# Or reproduce from the committed ingestion extracts:
uv run python -m supply_chain_opt_engine.reconciliation --source ingestion
```

The default upstream is the latest complete LightGBM run. Use
`--forecast-manifest artifacts/models/lightgbm/<run_id>/manifest.json` to pin it.
Upstream checksums, selected IDs, target labels and reconstructed median forecasts
must match the current source. The primary reconciliation method defaults to
MinT-shrink as requested by the project brief; this does not imply that it has
the lowest measured error.

Bundles under `artifacts/forecasts/reconciliation/<run_id>/` contain:

| File | Content |
| --- | --- |
| `hierarchy_nodes.csv`, `hierarchy.npz`, `tags.json` | Stable node ordering, parents, summing matrix and tags |
| `predictions.parquet`, `latest_forecasts.parquet` | Base and reconciled centers, marginal quantiles and row keys |
| `latest_leaf_forecasts.parquet` | Primary-method marginal leaf inputs for the next optimization step |
| `*_fit.npz` | Shrinkage covariance, projection matrices, training residuals and their dates |
| `fold_4_*_samples.npz`, `latest_*_samples.npz` | Full float64 coherent joint paths, node IDs and dates |
| `series_metrics.parquet`, `fold_metrics.csv`, `summary_metrics.csv` | Accuracy and probabilistic scores at every level |
| `coherence.csv` | Point/path errors, quantile non-additivity, clipping and fallback diagnostics |
| `runtimes.csv`, `manifest.json` | Execution measurements, protocol, source lineage and checksums |

NumPy archives load with `allow_pickle=False`. Coherence checks compare each
point/path with the same hierarchy applied to its leaves, with a declared
numerical tolerance. Native MinT's low-rank structure and the sparse Bottom-Up
and Top-Down matrices avoid dense full-hierarchy multiplication for every path.

Implementation follows Nixtla's
[reconciliation methods](https://github.com/Nixtla/hierarchicalforecast/blob/main/hierarchicalforecast/methods.py).

## Measured reference run

Run `20261111T175055Z-ad5ca98d` reconciled all four seasonal windows and the latest
origin in approximately 88 seconds across its five recorded phases. It produced
274,624 historical rows, 68,656 latest rows and 16,800 primary-method leaf rows.
The six saved holdout/production sample arrays each have shape `(613, 28, 501)`.
All 22 artifact checksums and four implementation checksums were independently
verified. Full-matrix coherence errors were below `7e-12`; saved sample quantiles
matched the forecast exports exactly. No Top-Down, MinT or ETS interval fallback
was used in this real-data run.

| Holdout level | Base WMAPE | Bottom-Up | Top-Down | MinT-shrink |
| --- | --- | --- | --- | --- |
| Total | 12.60% | 19.22% | 12.60% | 15.09% |
| Store | 14.19% | 19.36% | 16.92% | 15.49% |
| Store-Category | 20.08% | 22.09% | 22.52% | 20.18% |
| Store-Item | 59.20% | 59.20% | 73.82% | 61.08% |

MinT improves aggregate WMAPE relative to Bottom-Up but worsens WMAPE relative
to the independent base at all levels on the holdout. Leaf coverage is 84.46%,
compared with 89.55% for the base and Bottom-Up. All-fold leaf MinT WMAPE is
55.76% and coverage is 84.89%; those figures include the upstream tuning folds.
No expected 2–5% gain is substituted for the measured results.

MinT clipped 1,268 of 16,800 holdout point leaves and 897,355 of 8,416,800
holdout sample leaves before reaggregation. The manifest and coherence table
preserve this repair's frequency. The coherent point is distinct from marginal
P50: holdout MinT marginal medians have a maximum additivity difference of
524.74 units. This is consistent with coherent joint samples and must not be
reported as a failure of path coherence.

[Portable evidence](benchmarks/README.md) includes the unmodified manifest,
level/fold metrics, coherence and runtime tables, node metadata, and independent
validation report. The regression suite passed 84 tests; five live PostgreSQL
tests skipped because `TEST_DATABASE_URL` was unset. Tests also exercise
near-perfect ETS fits with undefined native intervals and verify the documented
training-error fallback.
