# Statistical baseline backtest

Step 3 evaluates SeasonalNaive with lag 7, AutoETS with weekly seasonality and
automatic model selection, and TSB with fixed smoothing parameters. The runner
uses StatsForecast. All models receive the same recorded-sales panel and the
same forecast origins. There is no random train/test split.

## Frozen seasonal windows

The origin is the last observed day. Training expands from the start of the
scoped history through the origin, inclusive.

| Fold | Origin | Validation dates | Season | Role |
| --- | --- | --- | --- | --- |
| 1 | 2015-02-28 | 2015-03-01 to 2015-03-28 | Spring | Tuning |
| 2 | 2015-05-31 | 2015-06-01 to 2015-06-28 | Summer | Tuning |
| 3 | 2015-08-16 | 2015-08-17 to 2015-09-13 | Back to school | Tuning |
| 4 | 2015-11-29 | 2015-11-30 to 2015-12-27 | Winter holidays | Holdout |

These dates are recorded in `configs/default.yaml` before model tuning. Selection
provenance is checked against the first origin. An ingestion run that selected
series using later sales cannot enter this backtest, even if the current config
claims an earlier selection cutoff. Overlapping windows, missing dates, missing
series, or insufficient training history fail before any fitting begins.

Baseline hyperparameters are fixed. TSB uses `alpha_d=0.1` and `alpha_p=0.1`;
AutoETS selects its exponential-smoothing specification using training data.
Later LightGBM tuning must use folds 1 to 3 only. Fold 4 is evaluated with frozen
choices. Saved summaries distinguish tuning folds, holdout, and all four folds.

## Recorded-sales policy

Models fit the recorded daily sales, including observed zeros on unlisted days.
Those rows are preserved, rather than silently removing dates or imputing latent
demand. Availability flags are retained in predictions and training diagnostics.
`listed_training_days` and `unlisted_training_days` show each series' exposure.

This is an observed-sales benchmark. It does not reconstruct sales lost to
stockouts or estimate demand before product introduction. Unlike the SQL rolling
features, the statistical models consume the full daily sales grid, including
recorded unlisted zeros. Both approaches preserve the distinction between listing
status and missing observations. A missing sales date remains an error.

## Probabilistic extension

All three point models receive StatsForecast `ConformalIntervals` with two
non-overlapping 28-day calibration windows nested inside each fold's training
history. The smallest inner training segment must have at least three weekly
cycles. Calibration never sees the outer validation window.

The `conformal_distribution` method constructs symmetric forecast distributions
from absolute calibration errors. We export P05, P10, P50, P80, P90, and P95.
The point forecast is the center and P50 of that constructed distribution; this
does not establish that it is the true conditional median of retail demand.
Bounds and point estimates are clipped at zero for nonnegative unit demand, with
clipping counts recorded. Quantile ordering and finite predictions are checked.
There is no silent substitution of another model on fitting failures.

With only two calibration windows, horizon-specific interval estimates have
limited support. Serial dependence and changing demand also affect coverage.
Measured 90% coverage is reported without claiming a guaranteed service level or
formal calibration for this dataset. Increase `conformal_windows` only using the
allowed tuning folds and repeat the frozen evaluation protocol.

## Metric definitions

- WMAPE is the sum of absolute point-forecast errors divided by actual units.
  Aggregate WMAPE pools errors and units, rather than averaging percentages.
  Errors on zero-demand series still contribute to the aggregate numerator.
- Per-series RMSSE divides validation mean squared error by the mean squared
  lag-7 difference from that fold's training history, then takes the square root.
  The reported aggregate is the equally weighted mean of defined series-fold
  RMSSE values. The scale uses recorded training sales, including unlisted zeros.
- Pinball loss is the mean of `max(q * error, (q - 1) * error)` at each exported
  quantile. `pinball_mean` equally averages those six quantile scores. Aggregate
  pinball losses are weighted by the number of forecast observations.
- Central 90% coverage counts actual observations inside P05 to P95, including
  both boundaries. Aggregate coverage is weighted by observation count.

A series with zero actual units has undefined WMAPE. A zero or absent RMSSE scale
has undefined RMSSE. Those values remain missing, and their counts are reported.
No epsilon or arbitrary denominator replaces them. All-fold summaries pool the
four validation windows; tuning summaries use only the first three. Per-series
and per-fold scores remain available for inspection.

## Run and inspect

Read the running PostgreSQL database using `.env`:

```bash
uv run python -m supply_chain_opt_engine.baseline
```

Or reproduce the benchmark from the checksummed CSVs that ingestion committed:

```bash
uv run python -m supply_chain_opt_engine.baseline --source ingestion
```

The latter uses the latest ingestion manifest with status `loaded`. An explicit
`--ingestion-manifest data/interim/<run_id>/manifest.json` selects another committed
run. It verifies the selected-ID and sales-artifact checksums before reading the
panel. The PostgreSQL source checks series membership, row count, and date bounds
against the same selection provenance. A content fingerprint records the actual
sales snapshot, so a changed database snapshot can be distinguished from the
original extract. Connection credentials never appear in artifacts.

Each run writes an atomic bundle to `artifacts/metrics/baselines/<run_id>/`:

| File | Content |
| --- | --- |
| `predictions.parquet` | Actual sales, point and quantile forecasts, availability, model, origin, horizon, fold, clipping flags |
| `training_scales.parquet` | Training-only RMSSE scales, date ranges, listing counts |
| `series_metrics.parquet` | Per-series, model and fold scores with denominator diagnostics |
| `fold_metrics.csv` | One row per fold and model |
| `summary_metrics.csv` | Separate all-fold, tuning-only and holdout metrics |
| `runtimes.csv` | Fit, calibration and prediction time per model and fold |
| `manifest.json` | Frozen dates, parameters, source provenance, scoring policies, versions and file/code hashes |

The baseline does not require the SQL feature export; univariate models read raw
sales. It does not create replenishment schedules or report simulated inventory
costs. Those require the later inventory-policy implementation.

## Measured reference run

The complete 600-series run generated 201,600 forecast observations across four
folds and three models. AutoETS achieved 60.12% four-fold WMAPE and 65.04% holdout
WMAPE. Seasonal Naive scored 72.93% and 83.20%, while TSB scored 61.59% and 69.08%.
TSB had the lowest four-fold mean RMSSE at 0.764, compared with 0.788 for AutoETS.
The nominal 90% intervals covered approximately 65% of observations for all three
models. Their two-window calibration is insufficient to claim reliable 90%
coverage in this setting. No parameters were changed after examining the holdout.

AutoETS took about 649.6 seconds in total for fitting, nested calibration and
prediction across four folds, using two workers. Seasonal Naive and TSB each took
about one second. The run used single-thread BLAS and Numba settings. Package JIT
caches may have been warmed by earlier pilot runs; timings are local measurements.
The [reference evidence](benchmarks/README.md) preserves exact scores, runtime
context, dependency versions and source/code hashes.

An independent NumPy check recomputed every RMSSE scale from pre-origin raw
sales, matched all saved target labels to the ingestion source, checked every
forecast grid and artifact hash, and verified pooled WMAPE, coverage, and pinball
losses against the summary CSV. Unit tests separately verify RMSSE aggregation and
undefined-denominator behavior.

Tests cover fold boundaries, leakage, missing daily observations, training-only
scales, undefined denominators, quantile ordering, actual StatsForecast models,
checksummed data sources, and artifact round trips. Implementation follows
Nixtla's [model definitions](https://github.com/Nixtla/statsforecast/blob/main/python/statsforecast/models.py)
and [rolling-origin evaluation](https://nixtla.github.io/statsforecast/docs/tutorials/crossvalidation.html).
