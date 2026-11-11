# Dash planning workbench

Step 7 provides a local, two-tab application backed by the validated forecast and
inventory artifacts. It does not query PostgreSQL or fit forecasting models when
users change chart selections.

```bash
uv sync --extra dashboard
uv run --extra dashboard python -m app.dashboard
```

Open [localhost:8050](http://127.0.0.1:8050). Start from the repository root.
`DASH_HOST`, `DASH_PORT` and `PROJECT_CONFIG` can be set in `.env`; command-line
`--host`, `--port` and `--root` override the corresponding launch defaults. Debug
mode and the development reloader are disabled. The default host is localhost.

Complete pipeline steps 1 through 6 first. With missing or invalid source data,
the app shows an actionable empty state rather than fabricated sample results.
Restart the application after producing a new forecast snapshot.

## Forecast explorer

The explorer opens on the latest saved origin. All four historical folds are
also available. Select a store, then a category, then an item. Selecting a higher
level resets dependent controls. The selections map to actual saved Total,
Store, Store-Category and Store-Item nodes; aggregate quantiles are never created
by adding leaf quantiles.

The chart shows 42 days of history, a 28-day coherent point forecast, and P10–P95
and P10–P90 uncertainty ranges. Historical selections also show their realized
validation demand. These later observations are displayed for evaluation and
never used by the planning inputs. Latest-origin forecasts have no future actuals.

P10–P95 has nominal coverage of 85%. The coherent center is distinct from the
marginal median. The daily table includes both center and P50 alongside P10,
P90 and P95. The footer identifies the forecast origin, node and reconciliation
run.

SHAP bars show model-wide mean absolute SHAP values from the saved production
or holdout P50 model. They explain associations, not causal effects or an
individual item's particular forecast. These are upstream global-model explanations;
they do not attribute the later MinT adjustment. Earlier tuning folds explicitly show
that no SHAP sample was saved; they do not inherit a later model's explanation.

## Replenishment simulator

Choose the historical holdout or the latest-origin scenario, then set:

- Demand quantile from the six available forecasts, P05, P10, P50, P80, P90, P95.
- Store capacity from 10% to 150% of the common synthetic baseline.
- Supplier lead-time delay from zero to four days.

The quantile slider uses six discrete positions. It never interpolates forecasts
that were not produced upstream. The UI accepts Dash's integer-valued float
slider events and validates each position before mapping it to a quantile.

Press **Run optimizer** to submit a background scenario. Status progresses through
pending, running, completed or failed. Controls and duplicate submissions are
disabled while the matching job is active. The forecast explorer remains usable.
A one-second poll reads persisted job status; it does not repeatedly redraw an
unchanged result.

Each result is tied to its exact phase, full configuration, upstream manifests
and optimizer implementation hashes. Identical requests reuse completed results
or the same active job. Validated reference runs can appear immediately with a
**Saved reference** label. Changing any scenario control clears the displayed
result and disables download until the new settings have a matching result.

The window selector switches between the 28-day decision window and the complete
window including the common 14-day tail. It changes reporting, not the solved
schedule. See [inventory definitions](optimization.md) for the objective,
lead-time convention, initial stock, pipeline, capacity and terminal treatment.

## Results and exports

The four main positions show total cost, fulfillment, average daily inventory
value, and purchase-order download with the number of new orders. Supporting
views show inventory and receipt trajectories, cost components, certified solver
gap, storage feasibility and policy comparisons.

Historical holdout results are labeled **Realized fill rate**. Latest-origin
results are labeled **Projected fulfillment** and explicitly describe a
forecast-demand replay. They do not claim an observed future service level.

The default cost-based plan selects zero new orders and serves only 23.06% of
28-day holdout demand under the original synthetic economics. This limitation
is shown next to the KPIs. P90 demand does not impose a minimum service constraint.
The UI does not present the low cost as evidence of a useful operating policy.
Comparison policies that exceed realized storage are marked. Their scores remain
diagnostic, and a primary result with a capacity failure cannot be downloaded
for the affected reporting window.

CSV downloads contain only the MILP policy, with integer quantities, placement
and arrival dates, unit costs, purchase values, fixed fees and scenario IDs.
Zero new orders produce a valid CSV with headers and no data rows. Failed,
unverified and stale scenarios cannot be downloaded.

## Background execution and persistence

`dashboard_jobs.py` launches a separate Python worker process. Files under
`artifacts/dashboard/jobs/` persist requests, status and logs. Atomic writes and
file locks deduplicate requests across browser sessions. A shared worker lock
serializes optimizer execution on the local host. Waiting jobs stay pending and
do not occupy a Dash request worker. A stopped worker is detected and reported
as failed; the same scenario can then be retried.

The worker calls the existing inventory workflow with pinned reconciliation and
baseline manifests. Result bundles and synthetic inputs retain the Step 6
locations and audit format. Failed solves produce no downloadable result.
Starting or importing the application never starts an optimizer job. Jobs already
submitted continue independently of tab navigation or browser closure.

This queue targets the project's local Linux environment. It uses filesystem
locks and a single shared filesystem; it is not a distributed worker service.
The existing roughly 30–40 seconds of recorded solve time is shown honestly.
The app does not promise the brief's unachieved 1–2 second scenario target.

## Implementation and checks

- `app/dashboard.py`: layout, filters, callbacks, source-bound result selection and download.
- `app/presentation.py`: native Plotly charts, tables and result labels.
- `app/assets/`: responsive warehouse palette, locally served Lato fonts and hidden-tab chart resize handling.
- `dashboard_data.py`: pinned, checksummed forecast snapshots and validated result views.
- `dashboard_jobs.py`: persisted requests, deduplication, worker lifecycle and failure handling.
- `tests/unit/test_dashboard.py`: real Flask/Dash callback tests and worker-state fixtures.

The layout follows the complete forecast and simulator concepts generated for
this step. The implementation uses real saved values in place of illustrative
concept values. It also adds the explicit reporting-window selector, source
provenance and uncertainty/feasibility explanations needed to interpret the data.
Charts, controls, tables and text are native components, not screenshots.

The callback regression suite covers hierarchy resets, invalid selections,
source checksums, model-wide SHAP scope, stale-result protection, future-versus-
realized labeling, zero-order CSVs, duplicate jobs, active-control locking,
worker completion/failure, retry and stopped-worker detection. Whole-number
float slider events have a dedicated regression test.

## Verified delivery

The regression suite passed 109 tests, with five live PostgreSQL tests skipped
because `TEST_DATABASE_URL` was unset. Lint, formatting, Python compilation and
JavaScript syntax checks passed. A time-limited feasible result with no finite
lower bound reports an undefined certified gap rather than inventing zero gap.

The Chrome/Playwright flow passed 16 checks on the real M5 artifacts: page
identity, hierarchy drill-down, exact saved holdout KPIs, CSV download, projected
future labels, stale-result clearing, background completion, active-control
locking, use of the forecast tab during a solve, infeasible-scenario handling,
download rejection and mobile overflow. No application console errors remained.
A separate visual pass checked the actual Plotly canvas bounds and both tabs on
desktop and mobile. Screenshots were compared with both design concepts.

The browser tests ran through the normal host terminal because the agent's
sandbox blocked sockets and Chrome, and the connected browser host was
unavailable. This was rendered application verification, not a static layout
substitute. Chrome was tested; other browser engines remain untested.

The review checked navigation and copy, the green/white palette, locally served
humanist typography, chart/table proportions, tab resizing, and the mobile
stack. Material fixes included discrete keyboard slider steps, handling numeric
float events, removing legend crowding and fitting charts initialized in hidden
tabs. Real values and complete policy comparisons intentionally replace the
concepts' illustrative data. The extra reporting-window selector and explicit
feasibility labels preserve the inventory evaluation's meaning.

The final visual pass used the concepts' 1487×1058 desktop canvas and a 390×844
mobile viewport. The earlier full functional run also covered 1440×1000. Both
concepts and final screenshots were inspected directly; no material layout
mismatch remained after the measured-canvas correction and mobile cleanup.
