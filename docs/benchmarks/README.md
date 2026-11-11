# Reference forecast evidence

These small files preserve the measured forecasting, reconciliation and inventory benchmarks without committing raw
M5 sales, full prediction panels, or fitted model data.

| File | Source in the complete local run |
| --- | --- |
| [baseline_summary.csv](baseline_summary.csv) | `summary_metrics.csv` |
| [baseline_folds.csv](baseline_folds.csv) | `fold_metrics.csv` |
| [baseline_runtimes.csv](baseline_runtimes.csv) | `runtimes.csv` |
| [baseline_run.json](baseline_run.json) | Unmodified run `manifest.json` |
| [baseline_validation.json](baseline_validation.json) | Independent validation report and runtime context |

The run ID is `20261111T162554Z-2f637d4b`. Its full local bundle is under
`artifacts/metrics/baselines/`. The source is the checksummed sales extract from
committed ingestion run `20261111T145941Z-91709646`. The run manifest records the
frozen dates, selection cutoff, model settings, metric policies, package versions,
code hashes, source fingerprint, and hashes for all generated artifacts.

Summary metrics named `wmape` and `coverage_90` are fractions. Multiply by 100 for
percentages. `rmsse` is the equal-series-fold average among defined scores.
`pinball_mean` averages the six exported quantile losses. Runtimes are seconds.
The `holdout` summary contains fold 4 only; `tuning_folds` contains folds 1 to 3.

Reproduce the protocol after ingesting the same scope:

```bash
uv run python -m supply_chain_opt_engine.baseline --source ingestion
```

See [baseline definitions](../baselines.md) for assumptions and limitations.

## Global LightGBM reference

Run `20261111T165106Z-52250620` used the same selected series and seasonal folds.
Its full local bundle is under `artifacts/models/lightgbm/`. Historical forecasts,
the production forecast panel, all native model files, and row-level SHAP values
remain in that ignored artifact directory.

| File | Source in the complete local run |
| --- | --- |
| [lightgbm_summary.csv](lightgbm_summary.csv) | `summary_metrics.csv` |
| [lightgbm_folds.csv](lightgbm_folds.csv) | `fold_metrics.csv` |
| [lightgbm_runtimes.csv](lightgbm_runtimes.csv) | `runtimes.csv` |
| [lightgbm_trials.csv](lightgbm_trials.csv) | `optuna_trials.csv` |
| [lightgbm_run.json](lightgbm_run.json) | Unmodified model-run `manifest.json` |
| [lightgbm_validation.json](lightgbm_validation.json) | Independent validation report |
| [lightgbm_holdout_shap_importance.csv](lightgbm_holdout_shap_importance.csv) | Frozen holdout median-model SHAP importance |
| [lightgbm_latest_shap_importance.csv](lightgbm_latest_shap_importance.csv) | Production median-model SHAP importance |

Optuna trial numbers in the CSV start at zero. The run contains 23 completed and
7 pruned trials; the study receives only folds 1 to 3. Use the holdout summary
for an independent comparison. Four-fold metrics also include the tuning folds.

Reproduce the workflow:

```bash
uv run python -m supply_chain_opt_engine.forecasting --source ingestion
```

See [forecasting definitions](../forecasting.md) for the direct design, source
parity, calibration diagnostics, saved-model interface, and repair policy.

## Hierarchical reconciliation reference

Run `20261111T175055Z-ad5ca98d` consumes the frozen LightGBM run above and adds
independent aggregate AutoETS forecasts. Its full local bundle is under
`artifacts/forecasts/reconciliation/`. Full point panels, fitted covariance,
projections and joint paths remain in that ignored artifact directory.

| File | Source in the complete local run |
| --- | --- |
| [reconciliation_summary.csv](reconciliation_summary.csv) | Level/model summaries for all folds, tuning folds and holdout |
| [reconciliation_folds.csv](reconciliation_folds.csv) | Per-fold level/model metrics |
| [reconciliation_coherence.csv](reconciliation_coherence.csv) | Coherence, clipping, residual windows and fallback diagnostics |
| [reconciliation_runtimes.csv](reconciliation_runtimes.csv) | Recorded execution times |
| [reconciliation_nodes.csv](reconciliation_nodes.csv) | Stable node indices, levels and parent metadata |
| [reconciliation_run.json](reconciliation_run.json) | Unmodified run manifest with source/artifact checksums |
| [reconciliation_validation.json](reconciliation_validation.json) | Independent matrix, label, quantile, lineage and metric checks |

WMAPE uses the coherent `point` center; pinball and coverage use sample marginal
quantiles, whose medians need not equal that center. MinT worsens holdout leaf
WMAPE from 59.20% to 61.08% and coverage from 89.55% to 84.46%. All methods are
coherent; Bottom-Up leaves exactly preserve the upstream point/quantile outputs.
The independent verifier uses full summing-matrix multiplication and finds
point/path differences below `7e-12`.

```bash
uv run python -m supply_chain_opt_engine.reconciliation --source ingestion
```

See [reconciliation definitions](../reconciliation.md) for projection repairs,
training-only residuals, copula sampling and measured limitations.

## Inventory optimization reference

Run `20261111T210441Z-d6b41773` planned the same 600 leaves for holdout and latest
origins. It used reconciled P90, a 28-day ordering window, a common 14-day no-order
tail and the original item economics. Full traces, schedules, purchase exports
and logs remain under `artifacts/orders/optimization/`; scenario side inputs are
under `data/synthetic/inventory/`.

| File | Content |
| --- | --- |
| [optimization_run.json](optimization_run.json) | Unmodified manifest with lineage, item/store bounds and code/artifact checksums |
| [optimization_metrics.csv](optimization_metrics.csv) | Forecast/realized costs, fill, capital and capacity flags for both windows |
| [optimization_solver.csv](optimization_solver.csv) | Store-level objectives, lower bounds, gaps, timings and strategy |
| [optimization_scenarios.csv](optimization_scenarios.csv) | Aggregate initial state, pipeline and store capacities |
| [optimization_validation.json](optimization_validation.json) | Independent saved-evidence replay and KPI verification |

All rates and gaps are fractions; costs are USD and volumes are m³. The scope
`decision_window` covers 28 days, while `including_tail` covers 42. Only holdout
has realized scores. `capacity_feasible` is evaluated for the corresponding
window. The global P90 rule breaches capacity during the tail; that full-window
replay is excluded from savings comparisons.

The MILP picks zero new orders for both phases. Its holdout fill is only 23.06%
over 28 days and 15.66% including the tail. This is a result of the specified
synthetic costs and missing service floor, not evidence of successful service
optimization. The combined solver gaps are 0.24% and 0.59%; reported low costs
must be read together with fulfillment. No reference savings percentage from
the project brief has been substituted for a measurement.

Reproduce and verify:

```bash
uv run python -m supply_chain_opt_engine.optimization
uv run python scripts/validate_inventory_run.py \
  artifacts/orders/optimization/<run_id>/manifest.json
```

See [optimization definitions](../optimization.md) for decision timing, tail
handling, policy rules, solver acceptance and evaluation limits.
