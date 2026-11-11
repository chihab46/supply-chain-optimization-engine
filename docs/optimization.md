# Inventory replenishment and policy evaluation

Step 6 builds a Pyomo MILP, solves it with HiGHS, exports validated purchase
orders, and compares frozen schedules against realized holdout sales. It also
creates a latest-origin plan. The dashboard remains the next stage.

## Run the workflow

```bash
uv run python -m supply_chain_opt_engine.optimization
# Limit execution to one phase:
uv run python -m supply_chain_opt_engine.optimization --phase holdout
uv run python -m supply_chain_opt_engine.optimization --phase latest
```

The default runs both phases. Pin upstream evidence with
`--reconciliation-manifest artifacts/forecasts/reconciliation/<run_id>/manifest.json`
and `--baseline-manifest artifacts/metrics/baselines/<run_id>/manifest.json`.
This stage reads checksummed extracts from the committed PostgreSQL ingestion;
it does not need a running database. It verifies reconciliation, global-model,
statistical-baseline and ingestion lineage before planning.

Settings live in `configs/default.yaml`. The demand quantile defaults to P90;
only the six exported quantiles are accepted. Capacity multipliers must be
positive. Lead-time shocks range from zero to four days and delay both new orders
and the existing pipeline. The default HiGHS solve budget is 60 seconds per
phase, shared across stores, with a relative MIP-gap target of 1% and one thread.
Model construction, validation, baseline planning and file writing add overhead.
Recorded phase timings exclude upstream source checks and scenario generation.
The Python API accepts `OptimizationConfig` for scenario execution.

## Scenario inputs

Each phase saves explicit side inputs under
`data/synthetic/inventory/<run_id>/<phase>/`. Costs and MOQ come from the original
item-level economics artifact, shared across stores. Starting stock, receipts
and capacity use only the 56 observed days ending at the forecast origin.

- Starting stock is the ceiling of mean demand times baseline lead time times a
  seeded multiplier between 0.75 and 1.25.
- One synthetic pre-origin pipeline order covers between 0.5 and 1 times mean
  demand times baseline lead time. Its arrival falls within the baseline lead
  time; a scenario shock delays that arrival.
- Baseline store capacity is 1.25 times the greater of initial stock volume and
  `sum((max(mean, daily P90) * (baseline lead + 7) + MOQ) * unit volume)`.
  The scenario capacity multiplier scales that fixed baseline.

Random draws are stable per seed, origin and series ID. Neither starting stock
nor baseline capacity changes when the forecast quantile or lead-time shock
changes. The manifest records units, generation rules, the training window and
checksums. These are reproducible scenario assumptions, not Walmart facts.

## Model and physical accounting

Orders `O[i,t]` are nonnegative integers. Binary triggers enforce MOQ for every
positive order. Inventory and lost sales are continuous nonnegative units because
forecast quantiles can be fractional. For every item and day:

```text
arrivals[t] = pre-origin pipeline[t] + O[t - lead_time]
I[t] - U[t] = I[t-1] + arrivals[t] - forecast_demand[t]
0 <= U[t] <= forecast_demand[t]
MOQ * trigger[t] <= O[t] <= finite_bound[t] * trigger[t]
sum(volume[i] * I[i,t]) <= store_capacity
```

Day one uses the saved initial stock. Orders placed on a day arrive before demand
on the day that is `lead_time` days later. A binary shortage disjunction prevents
positive ending stock and positive lost sales for the same item on the same day.
Receipts and sales cannot be rationed to create artificial future stock.

Order bounds come from remaining modeled demand, MOQ and storage on the arrival
day. Orders arriving after the modeled horizon are disabled. An order is also
fixed to zero when even eliminating all remaining lost-sales penalties cannot
pay its fixed ordering fee. With nonnegative holding costs and no terminal stock
minimum, this removes an economically dominated action.

The objective is fixed ordering fees plus daily holding costs plus per-unit
lost-sales penalties. Purchase costs appear in purchase-order values and inventory
capital, but are excluded from this objective as specified in the requirements.
There is no service-level constraint. A P90 demand input is a planning scenario;
it does not enforce a 90% realized fill rate.

## Ending the comparison

Each phase has 28 order-decision days followed by a shared 14-day tail. No policy
may place new orders during the tail. Existing orders and pipeline receipts still
arrive. Optimization uses a repetition of each input forecast's last seven days
for tail demand. The holdout evaluation uses the actual next 14 days for every
policy. Future actual sales never enter planning.

The baseline's last permitted review covers all remaining forecast tail demand.
This makes its response to the no-order period explicit. Ending stock receives
no salvage credit, and its quantity and value remain visible. The workflow
reports the original 28-day costs separately from the complete 42-day comparison.
Costs over the complete comparison include tail carrying costs and shortages.
A finite tail is still an assumption about the comparison boundary.

## Solve strategy and acceptance

Stores have no shared capacity or budget, so they can be solved separately.
Within a store, the runner first removes shared capacity and solves item
subproblems. Items with no economically admissible orders use their deterministic
zero-order trajectory. The sum of the item lower bounds is a valid lower bound
for the coupled store problem.

If the combined item plans satisfy shared capacity, their cost and the summed
bound certify the reported gap. Otherwise the runner solves the full coupled
store MILP. Unused time can refine item problems with the largest remaining bound
gaps. Tests cover both paths. The strategy changes computation, not the objective
or the shared-capacity acceptance rule.

HiGHS receives a feasible physical starting trajectory where one exists. Solutions
are loaded only after checking termination and a finite incumbent objective.
Time-limited incumbents remain labeled with their termination reason and gap.
A solver's `optimal` termination can mean optimal within configured tolerances;
the recorded lower bound and gap are retained. No incumbent means no purchase
order export for that failed phase.

After extraction, an independent physical replay checks integer quantities, MOQ,
arrival dates, work-conserving fulfillment, capacity and objective accounting.
The workflow never exports a schedule that fails this forecast-input validation.
The integration follows Pyomo's documented
[APPSI result handling](https://pyomo.readthedocs.io/en/6.8.1/api/pyomo.contrib.appsi.base.Results.html)
and [HiGHS interface](https://pyomo.readthedocs.io/en/stable/api/pyomo.contrib.appsi.solvers.highs.Highs.html).

## Policies and realized evaluation

All schedules are frozen at the origin. Their daily reviews operate on forecast
inventory, never on later realized demand. Evaluation replays those schedules
unchanged against the same observed sales and side inputs.

| Policy ID | Planning rule |
| --- | --- |
| `lightgbm_mint_milp` | Cost-minimizing MILP on reconciled leaf P90 by default |
| `lightgbm_p50_order_up_to` | Daily order-up-to using unreconciled global P50 |
| `lightgbm_p90_order_up_to` | Same rule using unreconciled global P90 |
| `seasonal_naive_p50_order_up_to` | Same rule using the statistical SeasonalNaive center |
| `autoets_static_safety_stock` | AutoETS center plus static `z * recent_daily_std * sqrt(lead + 1)` |

The normal multiplier `z` defaults to 1.28155. This safety-stock heuristic is not
an empirical coverage guarantee. Statistical policies use the saved holdout
backtest forecasts; latest-origin output includes the MILP and the two global
LightGBM policies.

Order-up-to uses current projected stock plus scheduled receipts as inventory
position. Its usual protection period is lead time plus one review day. Positive
orders round up to integer units and at least MOQ. A greedy admission step limits
orders to forecast-feasible future storage, resolving conflicts in stable series
order. The final review uses the remaining tail as described above.

Reported KPIs include fulfilled units divided by actual demand, holding,
stockout and fixed-order costs, their total, and mean daily ending inventory
value. Zero demand produces an undefined fill rate. M5 sales are a proxy for
demand; unobserved historical lost sales are unavailable.

A forecast-feasible schedule may exceed storage when actual demand is lower.
Realized replay preserves the physical units and exposes that capacity breach.
It does not silently discard stock or assume free external storage. Feasibility
flags and maximum excess volume apply to each reporting window. Cost-saving
comparisons exclude capacity-infeasible windows. Such replay KPIs are diagnostic
and must not be presented as feasible operational policy results.

## Saved evidence

Bundles under `artifacts/orders/optimization/<run_id>/` contain:

| File | Content |
| --- | --- |
| `manifest.json` | Config, source lineage, scenario checksums, versions, hardware and code hashes |
| `policy_metrics.csv` | Forecast and realized scores for the 28-day and complete windows |
| `<phase>/purchase_orders.csv` | Positive integer orders, placement/arrival dates, costs and unique scenario ID |
| `<phase>/order_schedules.npz` | Full schedules, including zero orders, and series ordering |
| `<phase>/inventory_trace.parquet` | Daily demand, orders, arrivals, fulfillment, stock, lost sales and costs |
| `<phase>/solver_diagnostics.json`, `*.log` | Incumbents, bounds, gaps, strategy and solver logs |

Scenario side inputs include `catalog.csv`, `starting_inventory.csv`,
`pipeline_receipts.csv`, `store_capacity.csv` and `manifest.json`. An empty
purchase-order CSV with headers is a valid output when no new order is selected.

Run the separate evidence verifier after completion:

```bash
uv run python scripts/validate_inventory_run.py \
  artifacts/orders/optimization/<run_id>/manifest.json
```

It uses the saved scenario files and row-level traces to replay accounting and
recompute costs, capacity flags and KPIs. It also verifies the purchase exports,
pre-origin state, common realized demand, solver bounds and all checksums.

## Measured reference run

Run `20261111T210441Z-d6b41773` completed both 600-leaf phases. Holdout took
41.35 seconds overall, including 33.09 seconds in solver calls. Latest-origin
execution took 36.74 seconds overall and 31.23 seconds in solver calls. The
combined incumbent objectives were $1,298.25 and $1,486.54 on their respective
forecast-demand inputs; the certified gaps were 0.24% and 0.59%. These are
forecast planning objectives, distinct from realized evaluation costs.

Every store's item solution satisfied shared capacity, so this reference run
used the verified item decomposition. Tests separately exercise the coupled
fallback. All main model/source checksums match the unmodified manifest.
Independent replay of every saved trajectory found zero inventory, fulfillment
or lost-sales discrepancies. It also matched 45,436 exported benchmark orders,
all costs, pre-origin state and solver bounds.

The MILP selects zero new orders for both origins. On the 28-day holdout, the
existing stock and receipts fulfill 23.06% of observed demand, with $118.50 in
holding costs, $391.17 in stockout costs and zero ordering fees, totaling $509.68
before rounding individual components. Complete-window fill falls to 15.66% and
cost rises to $810.73. The global P50 order-up-to baseline fulfills 73.97% over
28 days and 71.15% including the tail, costing $35,874.75 and $36,284.09.

The cost-based policy accepts large shortages because the original stockout
penalties are small relative to fixed ordering fees and carrying costs. The
implementation does not adjust economics or introduce a service constraint
using holdout results. Low cost here is not evidence of a useful operational
service policy. Positive-order analytical tests verify that the model does
order when doing so is economically favorable.

All forecast-input plans passed storage checks. The realized global P90 baseline
is feasible over the first 28 days but exceeds tail capacity by up to 130.34 m³.
Its complete-window scores are explicitly marked infeasible and excluded from
cost-saving comparisons. No inventory is silently discarded to hide this breach.
The MILP and other realized baseline replays remain within storage capacity.

The full regression suite passed 98 tests, with five live PostgreSQL tests
skipped because `TEST_DATABASE_URL` was unset. Lint, formatting and compilation
also passed. [Portable evidence](benchmarks/README.md) preserves the exact metrics,
solver bounds, scenario summaries, manifest and independent validation report.
