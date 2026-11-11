# SQL feature definitions

The implementation in [02_feature_view.sql](../sql/02_feature_view.sql) uses a
direct multi-step design. The forecast origin is the last observed demand day.
Each prediction row describes one store-item, origin, and future target date.
The horizon is a model input, alongside target-calendar features and demand
statistics anchored to the origin. This is a feature contract; model training
remains a later pipeline stage.

## Demand features

| Columns | Definition |
| --- | --- |
| `origin_lag_7`, `origin_lag_14`, `origin_lag_28` | Listed sales exactly 7, 14, or 28 calendar days before the origin |
| `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28` | Mean listed sales in the last 7, 14, or 28 calendar days, including the origin |
| `rolling_std_7`, `rolling_std_14`, `rolling_std_28` | Sample standard deviation over the same windows |
| `zero_sales_share_28` | Fraction of listed observations with zero sales during the last 28 calendar days |
| `available_days_7`, `available_days_14`, `available_days_28` | Number of listed observations contributing to the corresponding window |
| `origin_is_available` | Observed weekly price listing flag on the origin day |

For target date `origin + h`, the rolling window of length `w` spans
`target_date - h - w + 1` through `target_date - h`. Increasing the target horizon
therefore never introduces realized future demand. Lag column names explicitly
refer to the origin, rather than claiming to be target-relative lags.

`feature_daily_demand` creates a dense calendar grid before applying SQL windows.
It retains observed unlisted zeros separately from missing observations. Unlisted
days do not contribute to demand means, deviations, or zero shares. A lag pointing
at an unlisted day is NULL. An all-unlisted window has NULL statistics and zero
available-day counts. A standard deviation needs at least two listed observations.

The forecast function requires 29 complete observed days for every series through
the origin, enough to define lag 28. A missing observation or availability flag
raises an error. The historical demand view itself retains warm-up rows, with
incomplete rolling statistics left NULL. Feature generation never imputes demand
from future data.

## Price features

`feature_weekly_prices` first creates one row per series and calendar week.
Window length is measured in calendar weeks, including weeks with no listed
price. This prevents missing weeks from extending the reference window further
into the past.

| Column | Definition |
| --- | --- |
| `sell_price` | Latest listed weekly price available under the selected scenario |
| `price_mean_52w` | Mean observed price in the as-of week and previous 51 calendar weeks |
| `price_relativity` | Scenario selling price divided by that weekly reference |
| `price_momentum` | As-of week's listed price / previous calendar week's listed price - 1 |
| `price_observed_weeks_52w` | Number of observed prices in that reference window |
| `price_age_days` | Target date minus the first calendar day of the last observed price week |
| `price_carried_forward` | True when the as-of week is unlisted but a prior weekly price exists |
| `known_future_prices` | Explicit scenario flag repeated on all output rows |

By default, the as-of week is the origin's week. The same known price and history
apply to all forecast dates. `price_age_days` increases across the horizon. A
missing current-week price uses the latest earlier price; future prices never
backfill missing history. `price_carried_forward` describes missing listings at
the as-of week, rather than ordinary extrapolation from the origin to future days.

With `--known-future-prices`, the as-of week becomes the target date's week. This
assumes all supplied future price plans are already known at the origin, including
intermediate weeks used in their price references. The option is a declared
scenario assumption and must not be enabled silently in a backtest.

A weekly price is assumed known from the first calendar date of its week. Partial
52-week histories are allowed and expose their observation count. The reference
ignores missing prices without forward-filling its inputs. Momentum remains NULL
when either adjacent week's raw price is missing. Missing references or zero
denominators yield NULL rather than an arbitrary ratio.

## Calendar and event features

Calendar fields describe the target date. They include CA SNAP eligibility,
ISO weekday and week number, month, year, day of month, event names, national
holiday status, and a scheduled-event flag.

A national holiday is an event with M5 event type `National` in either event
slot. `days_until_next_holiday` includes the current date, giving zero on the
holiday itself. It remains NULL if the supplied calendar contains no later
national event. Religious, cultural, and sporting events remain available through
event names and the scheduled-event flag, but do not count as national holidays.
The calendar is treated as published information known in advance. M5 has no
complete promotion schedule; SNAP, scheduled events, and price signals are proxies,
not independently observed promotion labels.

## SQL objects and execution

- `feature_calendar` exposes target calendar fields and holiday countdowns.
- `feature_weekly_prices` exposes price history at the distinct-week grain.
- `feature_daily_demand` exposes historical origin-level demand statistics.
- `build_forecast_features(origin, horizon, known_future_prices)` returns a
  complete future panel for horizons between 1 and 28 days.
- `model_forecast_features` calls that function for the latest sales origin and
  28 days with the default price policy.

The historical demand view computes all origins with SQL windows. Single-origin
requests instead aggregate the required 29-day slice with identical date
boundaries. Small origin, price, and target-calendar snapshots are materialized
once before expanding the forecast horizon. Tests compare the bounded snapshot
with the historical window view to prevent definition drift.

These are views and a stable, read-only function, not cached feature tables. They
read current source data on each call. The export command installs them
transactionally and queries features and source counts in one read-only
repeatable snapshot. It validates keys, horizons, source-series cardinality, and
finite numeric values before writing an atomic Parquet bundle and manifest.

Historical training labels must be joined separately from `fact_daily_sales` by
store, item, and target date. Never pass those labels back into the feature query.
Choose training origins so every label date is inside its fold's training range.
The SQL objects intentionally do not create a training/holdout split.

## Verification

`tests/sql/` executes the actual SQL with PostgreSQL server binaries in an isolated
single-user cluster. It checks window edges, series partitioning, sample standard
deviations, zero shares, weekly references, missing-price behavior, holiday
countdowns, complete grids, installation reruns, and prediction invariance under
changes to post-origin demand and default-scenario future prices.

Live integration tests additionally exercise the Python query, repeatable-read
transaction, Parquet export, source metadata, and latest-origin default. They
require `TEST_DATABASE_URL` and create disposable schemas.

A full M5 check using the loaded ingestion CSVs in a temporary PostgreSQL 16
cluster produced 16,800 rows and 41 columns for 600 series at origin 2016-05-22.
Targets span 2016-05-23 through 2016-06-19. Demand and price features were complete;
event names and the holiday countdown retain expected missing values near the
end of the supplied calendar. The optimized install/query/CSV check took about
2.2 seconds on this machine, excluding source copying. This is a local
measurement, not a runtime guarantee. Its report is
`artifacts/metrics/feature_sql_validation.json`.

That check does not install views in the running Docker database. Run
`uv run python -m supply_chain_opt_engine.features` against the configured
database to install them and produce the versioned Parquet export.

The implementation uses PostgreSQL's documented
[window frames](https://www.postgresql.org/docs/16/functions-window.html) and
[stable function behavior](https://www.postgresql.org/docs/16/sql-createfunction.html).
