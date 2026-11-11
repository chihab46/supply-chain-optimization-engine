# Test plan

Ingestion tests cover panel size and hierarchy representation, selection leakage,
source-order independence, daily grids, availability, invalid values, duplicate
keys, metadata consistency, deterministic economics, audit bundles, and download
failure/archive handling.

Run `uv run pytest -m 'not integration'` for local tests. To exercise PostgreSQL,
set `TEST_DATABASE_URL` to a database where the test user may create schemas and
run `uv run pytest -m integration`. Each database test gets a unique schema and
removes it afterward. Database tests check COPY/upsert reruns, row counts,
transaction rollback after a late foreign-key failure, and rejection of changed
scope or economics. They skip when no test database is configured.

Feature tests in `unit/test_features.py` check panel validation and label exclusion.
`sql/` tests run actual feature SQL in a temporary PostgreSQL single-user cluster,
without network sockets. They require PostgreSQL server binaries and a non-root
user. They check demand windows and lags, weekly price histories, missing listings,
holiday countdowns, grid gaps, installation reruns, and future-data invariance.
They skip explicitly if local PostgreSQL is unavailable. The live feature
integration test also exercises the Parquet exporter and Python database adapter.

Statistical baseline tests exercise four seasonal splits, selection cutoff
provenance, complete grids, training-only RMSSE scales, undefined metrics, pooled
versus per-series aggregation, and asymmetric pinball penalties. Real StatsForecast
models are tested for nonnegative ordered quantiles and forecast invariance after
post-origin sales change. A four-fold fixture checks artifact hashes, Parquet
round trips, and separate tuning/holdout summaries.

Global forecasting tests check origin-bounded labels/features, invariance to
future sales and prices, a holdout-free tuning API, P50-preserving crossing
repair, an actual Optuna/MLForecast/LightGBM workflow, saved-model predictions,
artifact hashes, and SHAP additivity. PostgreSQL parity tests compare the offline
feature store with all SQL panel columns across missing-listing scenarios.

Reconciliation tests cover stable hierarchy indices, exact sums, learned Nixtla
projections, nonnegative reaggregation, seeded circular rank paths, zero-atom
quantile preservation, and non-additive marginal medians of coherent paths.
The end-to-end workflow verifies artifacts, samples, fixed Bottom-Up leaves,
fallbacks for degenerate fits, and invariance of covariance/proportions when
future demand changes.

Inventory tests compare a tiny MILP with exhaustive enumeration and verify
fractional demand, integer orders, MOQ, shared capacity, pipeline infeasibility,
physical fulfillment, economically dominated actions, zero demand, and lead-time
shocks. They cover both item decomposition and the coupled capacity fallback.
Solver status tests reject missing incumbents and preserve time-limited gaps.
Scenario tests prove post-origin demand cannot change state; workflow tests prove
realized demand cannot change frozen orders. Replay tests check costs, inventory
capital, late arrivals, final tail coverage and per-window storage flags.
`scripts/validate_inventory_run.py` independently audits full saved bundles.

Dashboard tests use actual Flask/Dash callback requests to verify source-backed
charts, hierarchy changes, explanatory scope, future/realized labels, whole-number
float slider events, stale-result clearing and CSV download gates. Worker tests
check deduplication, active-control locking, completion, failure, retry and stopped
process detection. Install `uv sync --extra dashboard` to include these tests;
otherwise the dashboard module skips explicitly.

Use `unit/` for small deterministic fixtures. Cover training-only series selection,
time cutoffs, feature availability at every horizon, train-only RMSSE scales,
zero-demand metric behavior, hierarchy sums, interval crossing, lead-time arrival
dates, MOQ triggers, and inventory accounting.

Use `integration/` with `@pytest.mark.integration` for PostgreSQL constraints,
repeatable ingestion, cardinality-preserving price joins, SQL feature cutoffs, and
round trips to artifact files. Integration tests require the local database.

Use tiny feasible and infeasible HiGHS examples to check capacity, integer orders,
initial inventory, pipeline receipts, lost-sales bounds, objective accounting,
and solver termination. Never accept a time limit without an incumbent solution.

Mark training or full-size solves with `@pytest.mark.slow`. Keep actual data,
credentials, and machine-dependent runtime thresholds out of unit tests.
