# Global quantile forecasting

Step 4 fits one global LightGBM model per quantile across all 600 store-item
series. The shared quantile contract is P05, P10, P50, P80, P90, and P95. P50 is
the point forecast used for WMAPE and RMSSE. Lower and upper quantiles provide
the forecast fan and central P05-P95 coverage measurement.

## Direct multi-step design

The model receives the forecast horizon as an input and predicts all 28 days
from a fixed origin. Its inputs are the Step 2 demand statistics, origin-relative
lags, price history, target-calendar fields, and categorical series, store, item,
department, category, and event identifiers. Demand inputs end at the origin.
Future prices are carried forward from the origin rather than read from realized
future weeks. The model does not recursively consume future realized sales.

Training examples use non-overlapping 28-day origin blocks. The default label
lookback is 365 days before the fit cutoff, with complete 28-day label blocks
ending no later than that cutoff. This yields 13 blocks and 218,400 examples for
each full-sized reference fold. Features can use older history for lag and price
reference windows. The exact origins and latest label date are saved per fold.
The last year of labels limits local training cost; older origin-available sales
still define the RMSSE denominator shared with the statistical baselines.

The feature encoder uses a fixed categorical schema from the selected panel and
the published calendar. These categories contain no target statistics. Numerical
features remain missing when undefined, which LightGBM handles directly. Models
fit recorded sales, including observed zeros on unlisted days, as in the baseline
comparison. SQL demand statistics continue to exclude unlisted days and expose
their available-observation counts.

MLForecast preprocesses the labeled panel with `static_features=[]`, then fits
the quantile estimators through its supported `fit_models` interface. The origin
features are supplied explicitly, so no additional recursive lag transforms are
added. At prediction time the fitted global estimators process all horizon rows
together. This is manual direct forecasting, rather than MLForecast's separate
model per horizon strategy. It requires six global estimators rather than 168
horizon-specific estimators.

## Features and source consistency

The default PostgreSQL path installs the feature SQL and reads sales, demand
views, weekly prices, calendar, and item metadata in one repeatable read-only
snapshot. Tuning uses that fixed snapshot throughout the study.

`--source ingestion` reads the exact checksummed extracts that ingestion
committed. A pandas implementation reproduces the SQL windows and price/calendar
definitions for this offline path. Real PostgreSQL parity tests compare all
41 panel columns for regular data, missing price weeks, unlisted periods, and
both supported price scenarios. Future-data mutation tests also check that
post-cutoff demand and future price changes cannot alter training features,
training labels, or origin-specific prediction inputs.

Input-table fingerprints, ingestion lineage, source code hashes, categories, and
the ordered feature list are stored with the model artifacts. The latest-origin
feature export from Step 2 is useful for inspection, but is not a historical
training dataset. Training constructs the required historical origins explicitly.

## Optuna and holdout discipline

The four frozen seasonal folds from Step 3 remain unchanged. Optuna receives
only folds 1, 2, and 3. Passing the holdout to the tuning function raises an error
before a study is created. Target labels for every training matrix must end by
its own fold's origin.

A seeded TPE search runs 30 trials by default. It tunes:

- `num_leaves` from 15 to 63.
- `min_child_samples` from 40 to 200 in steps of 20.
- `learning_rate` from 0.03 to 0.15 on a logarithmic scale.
- `colsample_bytree` from 0.6 to 1.0.

The objective is mean P50 pinball loss across the three equal-sized tuning
windows. This also minimizes their pooled absolute error, and therefore their
pooled WMAPE since actual demand totals are fixed. A median pruner can stop
unpromising trials after partial tuning-fold evaluation. Only completed trials
can supply the winning parameters.

The number of trees is fixed at 160, with two model threads and a 63-bin
histogram configuration. Trials run sequentially for seeded reproducibility.
The same winning tree settings are reused for the other quantiles. Their pinball
losses and coverage are evaluated, but the search does not separately optimize
upper-quantile calibration. All models use LightGBM's `quantile` objective and
their own `alpha`. There is no holdout-based early stopping.

After tuning, all six quantiles are fitted and scored at each frozen origin.
Fold 4 is evaluated with frozen parameters. Scores from folds 1 to 3 have been
used for selection and are validation results, not independent test estimates.
The all-fold summary includes those tuning folds. Use the separate holdout
summary for the strongest comparison with the statistical baselines.

Only after scoring, a production model is refitted at the latest observed origin
using the same frozen parameters. It may use observations newer than the holdout,
but cannot change the saved historical forecasts or their scores. Holdout and
production models are stored separately.

## Crossing repair and SHAP

Independent quantile models can cross. The declared repair policy clips P50 at
zero, sorts the two lower forecasts and caps them at P50, then sorts the three
upper forecasts and floors them at P50. This produces ordered nonnegative
quantiles while preserving the median model's point prediction. The policy is
fixed before tuning, and the artifact records how many rows were repaired.
Coverage and pinball scores use the repaired forecasts. This does not establish
a coherent joint distribution across hierarchy nodes; reconciliation is Step 5.

TreeSHAP explains the raw P50 model. The default sample contains each series'
first forecast day, up to 600 rows, for both holdout and production models.
Contributions plus the expected value must reconstruct the raw model prediction
within numerical tolerance. The diagnostic also stores the nonnegative reported
point forecast, which may differ after clipping. Other quantile repairs do not
change P50. Feature values, row keys, SHAP contributions, and mean absolute SHAP
importance are saved. These diagnostics describe model dependence and do not
prove causal price elasticity or a causal SNAP effect.

## Run and artifacts

```bash
# Read SQL views from the configured PostgreSQL database:
uv run python -m supply_chain_opt_engine.forecasting
# Or use the committed ingestion extracts without a live database:
uv run python -m supply_chain_opt_engine.forecasting --source ingestion
```

An optional `--ingestion-manifest` selects a committed run explicitly. Each study
has a persistent SQLite file beside the output bundles, so Optuna trial history
remains inspectable if a later phase fails. A completed bundle also includes a
consistent SQLite backup. The CLI starts a new seeded study for each run.

Bundles live under `artifacts/models/lightgbm/<run_id>/`:

| File or directory | Content |
| --- | --- |
| `holdout/`, `latest/` | Six native LightGBM model files and JSON feature/category metadata |
| `predictions.parquet` | Historical point/quantile predictions, actual labels, origins and repair flags |
| `latest_forecasts.parquet` | Latest-origin 28-day forecasts without realized labels |
| `training_scales.parquet`, `series_metrics.parquet` | Baseline-compatible scale and score diagnostics |
| `fold_metrics.csv`, `summary_metrics.csv` | Per-fold, tuning-only, holdout and all-fold scores |
| `optuna_trials.csv`, `study.sqlite3`, `best_params.json` | Full trial evidence and selected settings |
| `holdout_shap.parquet`, `latest_shap.parquet` | Row-level median-model explanations |
| `*_shap_importance.csv` | Mean absolute SHAP by feature |
| `runtimes.csv`, `manifest.json` | Phase measurements, frozen protocol, provenance and checksums |

Native model loading uses `learning.predict_saved(model_directory, panel)` with
the stored feature order and categorical schema. Artifacts contain no Python
pickle dependency. Model-fit and end-to-end tuning runtimes are separate costs.
The under-three-minute training goal applies to one global quantile-model set,
not the complete 30-trial seasonal search.

The implementation follows the supported
[MLForecast fitting interface](https://github.com/Nixtla/mlforecast/blob/main/mlforecast/forecast.py)
and LightGBM's [quantile parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html).

## Measured reference run

Run `20261111T165106Z-52250620` used all 600 series and the frozen 2015 seasonal
schedule. Of 30 Optuna trials, 23 completed and 7 were pruned. The selected
parameters were 59 leaves, 80 minimum child samples, learning rate approximately
0.07204, and column sampling approximately 0.69640.

| Scope | WMAPE | Mean RMSSE | Central 90% coverage | Mean pinball loss |
| --- | --- | --- | --- | --- |
| Tuning folds 1-3 | 54.10% | 0.761 | 88.35% | 1.079 |
| Untouched holdout | 59.20% | 0.726 | 89.55% | 0.984 |
| All four folds | 55.22% | 0.752 | 88.65% | 1.055 |

The holdout WMAPE is 8.99% lower than AutoETS's 65.04%. AutoETS holdout central-90%
coverage was 67.65%, compared with 89.55% for the direct quantile model. These are
forecast results; no inventory fill-rate or cost improvement is claimed yet.

Each training matrix had 218,400 examples and 38 encoded features. Frozen fold
training and prediction phases took approximately 46 to 56 seconds per quantile
set. The latest-origin refit phase took about 57 seconds including SHAP and model
export. Measurements used two model threads and single-thread BLAS settings on
this machine. Optuna tuning costs are recorded separately in trial durations.

The fixed nonnegativity/order repair changed 46,872 historical forecast rows.
This count includes both negative-value clipping and quantile-order adjustment,
so it must not be interpreted as a pure crossing rate. Raw P50 predictions are
stored separately. The largest median-model SHAP effects were recent rolling
sales and series/item identity. Price and calendar drivers remain inspectable in
the full importance tables, without asserting that they dominate this run.

Independent checks verified all artifact and executed-code hashes, the 30-trial
study's exclusion of fold 4, target-label dates and values, pooled scores, native
model reloads for holdout and production, 600-series SHAP additivity in each role,
and identical RMSSE denominators to the statistical benchmark. The full-data
offline feature panel also matched all 16,800 rows and 41 columns of PostgreSQL
output. Compact evidence is versioned under [benchmarks](benchmarks/README.md).
