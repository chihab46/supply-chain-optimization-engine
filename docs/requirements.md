# Project requirements

This document defines the implementation scope. It does not claim that the
pipeline, performance targets, or model accuracy have been achieved.

## Scope and decisions

- Forecast daily demand over 28 days for at most 600 store-item leaf series across
  CA_1, CA_2, CA_3 and FOODS, HOUSEHOLD, HOBBIES, using roughly 1,100 calendar days.
- Interpret "600 SKUs" as 600 store-item pairs to match the brief's 600 hierarchy
  leaves. Document the number of distinct items separately.
- Use Total -> Store -> Store-Category -> Store-Item. Department remains metadata;
  department forecasts would require an additional hierarchy level.
- Treat the initial capacity constraint as one backroom capacity per store. A
  shared warehouse and procurement budget need separate definitions and are
  outside the first model.
- Use Python 3.12, PostgreSQL 16, uv, the Nixtla libraries, LightGBM, Pyomo, HiGHS,
  and Dash. Keep stochastic metadata reproducible with seed 42.
- Treat demand as lost sales rather than carried backlog. Report assumptions
  about observed sales as a proxy for demand because M5 does not reveal all
  historical lost demand.

## Data and contracts

| Dataset | Grain / key | Required content |
| --- | --- | --- |
| `dim_calendar` | date | M5 day number, week, calendar fields, CA SNAP, events |
| `dim_prices` | store, item, week | Positive selling price |
| `dim_sku_economics` | item | Department, category, purchase/holding/stockout costs, volume, lead time, MOQ, fixed order fee |
| `fact_daily_sales` | store, item, date | Nonnegative integer sales, observed-price availability flag |
| Store scenario metadata | store | Capacity in m3, scenario identifier |
| Starting inventory | store, item, origin | Nonnegative on-hand units |
| Pipeline receipts | store, item, arrival date, origin | Units already ordered before the origin |
| Forecast panel | unique_id, ds, model, quantile, origin | Nonnegative forecast units, hierarchy level, fold |
| Purchase orders | store, item, order date, arrival date, scenario | Integer quantity, unit cost, fixed order fee |

The four SQL tables form the initial database. Scenario metadata, starting stock,
and pipeline receipts initially live in versioned Parquet/CSV artifacts under
`data/synthetic/`, with units and generation parameters recorded alongside them.
Do not assume zero starting stock silently. Generate economics once per item and
share them across stores. Lead times are 2, 3, or 5 days; stockout penalty is 3 to 8
times daily holding cost. Capacity and cost choices are synthetic assumptions,
not facts inferred from M5.

Ingestion must reject duplicate keys, invalid categories, negative sales, and
inconsistent item metadata. Preserve a complete daily grid for every selected
series. Distinguish zero sales from missing prices and pre-introduction periods.
Define availability before imputing any missing observation. Joins must preserve
the number of store-item-date rows. Upserts or staged transactions must make
repeated ingestion safe.

Select series from data available before the first evaluation origin. Stratify by
store/category and demand velocity/intermittency. Save the selected IDs and source
file checksums; do not rank using validation or holdout sales. Quantile comparisons
must use the same selected panel.

## Features and leakage prevention

`sql/02_feature_view.sql` must implement demand lags 7/14/28, rolling means and
standard deviations 7/14/28, the 28-day zero-sales share, SNAP, scheduled events,
days until the next holiday, price relativity, and weekly price momentum.

Every feature has an availability date. Demand features for a prediction at
origin + k must use only observations at or before the origin. Choose either
direct horizon-specific features, with rolling windows shifted by the horizon,
or recursive updates using prior predictions. Shifting by one row while evaluating
28 future days against observed sales is leakage. SQL windows also need a complete
calendar because ROWS counts observations rather than calendar days.

Compute the 52-week price reference at the distinct-week grain before joining
back to daily sales. Future selling prices are allowed only under an explicit
known-price scenario. Otherwise carry forward the most recent known price and
record that assumption. Never backfill from future prices. Holidays and SNAP
flags can use the published calendar. Freeze missing-value and warm-up policies
inside each fold.

Acceptance requires fixtures proving that changing post-origin sales cannot
change any pre-origin feature or prediction input. Assert lag boundaries, price
join cardinality, and daily-grid completeness.

## Evaluation and forecasting

- Freeze four chronological, non-overlapping 28-day validation windows with
  adequate training history, covering spring, summer, back-to-school, and winter
  holidays within the available data. Persist exact dates before tuning. The
  inspected M5 calendar supports the frozen 2015 schedule in
  [baselines.md](baselines.md), now recorded in the configuration.
- Run SeasonalNaive with season length 7, AutoETS, and TSB. The brief's reference
  to Croston is satisfied by the explicitly requested TSB intermittent benchmark.
- Train global LightGBM quantile models through MLForecast for P05, P10, P50,
  P80, P90, and P95. P05 is added so the coverage metric can use a central 90%
  interval, P05 to P95. The requested P10 to P95 fan remains available and has
  nominal coverage of 85%.
- Tune num_leaves, min_child_samples, learning_rate, and colsample_bytree through
  Optuna on folds 1 to 3 only. Freeze preprocessing and hyperparameters before
  evaluating fold 4. Fit holdout training models using only pre-holdout data.
- Report WMAPE = sum(abs(y - prediction)) / sum(y), seasonal RMSSE using lag 7
  differences from each series' training history, pinball loss per quantile, and
  empirical central-90% interval coverage. Pinball and coverage are N/A for point
  baselines without a defined probabilistic extension.
- Specify RMSSE aggregation weights in the metric artifact. Flag zero-demand
  WMAPE and zero-scale RMSSE as undefined, report their counts, and document how
  they affect aggregation. Never silently substitute an arbitrary denominator.
- Extract SHAP outputs with feature names, model version, and explained sample
  scope. Save prediction runtime and training runtime separately.

Artifacts must include cutoff dates, selected IDs, seed, configuration, library
versions, train range, model, quantile, hierarchy level, and fold ID. Report the
four-fold summary and each fold separately. Do not advertise accuracy gains before
measuring them.

## Hierarchical reconciliation

Construct a summing matrix with stable leaf IDs and explicit level tags. Apply
Bottom-Up, Top-Down, and MinT with shrinkage. For Top-Down and MinT, generate base
forecasts at the required aggregate nodes. Estimate proportions and residual
covariance using training data only, with time-aligned residuals.

Validate aggregate forecasts against leaf sums within a documented numerical
tolerance. Report WMAPE at every level before and after reconciliation. Forecasts
may worsen at some levels; record that result faithfully.

Marginal quantiles do not generally add across children. Reconcile joint forecast
samples or paths and derive marginal quantiles from the reconciled samples.
Validate coherence on samples and point forecasts, and check marginal quantile
ordering, nonnegative demand, and coverage separately. Do not claim that separately
reconciled marginal quantiles form a coherent joint distribution. If the optimizer
uses leaf P90 demand, label it a quantile-demand planning scenario rather than a
guarantee of a 90% realized fill rate.

## Inventory model and simulation

Use nonnegative integer order quantities O and binary order triggers z. Ending
inventory I and lost sales U are nonnegative. For each store-item and day:

```text
arrivals[t] = known_pipeline_receipts[t] + O[t - lead_time]
I[t] = I[t-1] + arrivals[t] - demand[t] + U[t]
0 <= U[t] <= demand[t]
MOQ * z[t] <= O[t] <= order_upper_bound[t] * z[t]
sum(unit_volume[item] * I[store, item, t]) <= store_capacity[store]
```

Use starting on-hand inventory for the first day. References to orders before the
horizon contribute only through the known pipeline, never free decision variables.
Receipts arrive before that day's demand. Initial storage constrains end-of-day
inventory as specified in the brief. Any inbound staging limit is a separate
future requirement. Derive finite order bounds from feasible capacity and demand,
rather than choosing an arbitrary huge big-M.

Minimize fixed ordering fees plus daily holding costs plus per-unit lost-sales
penalties. Require work-conserving fulfillment: a day cannot end with both positive
inventory and unmet demand for the same item. Add a bounded binary disjunction or
an equivalent formulation; flow balance alone allows inventory rationing, especially
under terminal-stock constraints. Use the same fulfillment rule and cost definitions
in optimization and evaluation. Purchase cost is
recorded for working-capital and order exports; adding it to the objective is a
separate policy decision that must also change baseline accounting.

Handle end-of-horizon behavior explicitly. The first comparison must either use a
shared terminal-stock requirement or extend the simulation with a common tail
policy. Prevent orders that arrive outside the modeled horizon from creating
unearned benefits or misleading purchase-order exports.

Test balances, lead times including shocks, MOQ, order integrality, capacities,
zero-demand cases, and infeasibility. Enforce the solve time limit and MIP gap.
Inspect termination and feasibility before using a solution; a time-limited run
with an incumbent must be labeled with its remaining gap. Missing or infeasible
solutions must not become purchase orders.

Simulate MILP and baseline policies against the same actual holdout demand, initial
stock, receipts, storage, and economics. Define Order-Up-To Mean P50 as a median
forecast policy, since P50 is not generally the mean. Specify its review cadence,
protection period, and MOQ/capacity handling. Also define the AutoETS static safety
stock policy before comparing it. A rolling daily replan may use only demand
observed up to that day's decision time.

Report fill rate = fulfilled units / actual demanded units, daily average inventory
value = mean over days of sum(unit purchase cost * ending inventory), holding cost,
stockout cost, fixed order cost, and their total. Expose undefined fill rate when
actual demand is zero. Report cost savings relative to an explicitly named policy.

## Dashboard

The two tabs must provide hierarchy drill-down, history, reconciled forecast bands,
SHAP explanations, and scenario controls for supported quantiles, capacity as a
percentage of baseline, and lead-time shocks from +1 to +4 days, with zero as the
default. Quantile choices must match available trained models or use a documented
interpolation method. Capacity presets need to include feasible and restrictive
scenarios, with bounds chosen after inspecting synthetic volumes.

Run optimization in a background job. Show pending, running, completed, and failed
states, prevent duplicate submissions, and associate results with their scenario.
Display total cost, fill rate, average inventory capital, solver status, and a
downloadable purchase-order CSV. Forecast-only future scenarios cannot report an
observed fill rate without realized demand; label simulated estimates accordingly.

## Delivery and performance

The brief's training under 3 minutes and scenario solves around 1 to 2 seconds are
benchmark targets, not guarantees. Record hardware, thread counts, dataset size,
solver gap, cold/warm state, and runtimes before assessing them. The initial solver
time limit is 60 seconds so a difficult scenario returns a defined outcome.

Keep database credentials in environment variables and source data outside Git.
Use a locked dependency environment and reproducible artifact metadata. Validate
Python imports without loading data or contacting the database. Implement tests
with each pipeline stage according to [tests/README.md](../tests/README.md).

Completion requires measured baseline/model metrics, checked hierarchy coherence,
feasible solver outputs, a fair holdout policy comparison, functioning dashboard
interactions, and a README evaluation table populated from saved metric artifacts.
