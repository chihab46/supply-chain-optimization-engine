"""Local two-tab Dash planning workbench. Start with python -m app.dashboard."""

from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import asdict
from pathlib import Path

import yaml
from dash import Dash, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate
from dotenv import load_dotenv

from app.presentation import (
    blank_results,
    empty_figure,
    forecast_figure,
    forecast_table,
    result_view,
    scenario_results,
    shap_figure,
)
from supply_chain_opt_engine.dashboard_data import ALL, DashboardRepository
from supply_chain_opt_engine.dashboard_jobs import ACTIVE, JobManager, selected_config
from supply_chain_opt_engine.optimization import OptimizationConfig

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
QUANTILE_CHOICES = (0.05, 0.1, 0.5, 0.8, 0.9, 0.95)
SIMULATOR_SUBTITLE = "Test a replenishment plan against common inventory and capacity."
QUANTILE_NOTE = "P90 demand is a planning scenario, not a minimum fill rate."
TAIL_LABEL = "Including 14-day tail"
GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}


def quantile_at(position):
    if (
        isinstance(position, bool)
        or not isinstance(position, (int, float))
        or not math.isfinite(position)
        or position != int(position)
        or not 0 <= position < len(QUANTILE_CHOICES)
    ):
        raise ValueError("Choose one of the six demand quantiles.")
    return QUANTILE_CHOICES[int(position)]


def dropdown(id, options, value, **kwargs):
    return dcc.Dropdown(
        id=id, options=options, value=value, clearable=False, searchable=True, **kwargs
    )


def field(label, component):
    return html.Div([html.Label(label, htmlFor=component.id, className="field-label"), component])


def config_from_project(root):
    path = Path(os.environ.get("PROJECT_CONFIG", root / "configs/default.yaml"))
    if not path.is_absolute():
        path = root / path
    doc = yaml.safe_load(path.read_text())
    opts = {
        k: v
        for k, v in doc["optimization"].items()
        if k in OptimizationConfig.__dataclass_fields__ and k != "seed"
    }
    return OptimizationConfig(**opts, seed=doc["seed"])


def layout(repository):
    return html.Div(
        [
            dcc.Store(id="job-id", storage_type="session"),
            dcc.Store(id="job-state"),
            dcc.Interval(id="job-poll", interval=1000),
            dcc.Download(id="order-download"),
            html.Header(
                html.Div(
                    [
                        html.Div("Supply Chain Engine", className="brand"),
                        html.Div(
                            f"{len(repository.leaves):,} store-item series / California",
                            className="header-meta",
                        ),
                    ],
                    className="header-inner",
                ),
                className="app-header",
            ),
            html.Main(
                [
                    dcc.Tabs(
                        id="workspace-tab",
                        value="forecast",
                        className="workspace-tabs",
                        children=[
                            dcc.Tab(
                                label="Forecast explorer",
                                value="forecast",
                                className="workspace-tab",
                                selected_className="workspace-tab-selected",
                            ),
                            dcc.Tab(
                                label="Replenishment simulator",
                                value="scenario",
                                className="workspace-tab",
                                selected_className="workspace-tab-selected",
                            ),
                        ],
                    ),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.H1("Forecast explorer"),
                                    html.P("Trace demand from store totals to individual items."),
                                ],
                                className="view-heading",
                            ),
                            html.Div(
                                [
                                    field(
                                        "Forecast origin",
                                        dropdown(
                                            "forecast-origin",
                                            [
                                                {"label": v, "value": k}
                                                for k, v in repository.origin_labels.items()
                                            ],
                                            "latest",
                                        ),
                                    ),
                                    field(
                                        "Store",
                                        dropdown(
                                            "store-filter",
                                            [
                                                {"label": "All stores", "value": ALL},
                                                *[
                                                    {"label": s, "value": s}
                                                    for s in sorted(
                                                        repository.leaves.store_id.unique()
                                                    )
                                                ],
                                            ],
                                            ALL,
                                        ),
                                    ),
                                    field(
                                        "Category",
                                        dropdown(
                                            "category-filter",
                                            [{"label": "All categories", "value": ALL}],
                                            ALL,
                                            disabled=True,
                                        ),
                                    ),
                                    field(
                                        "Item",
                                        dropdown(
                                            "item-filter",
                                            [{"label": "All items", "value": ALL}],
                                            ALL,
                                            disabled=True,
                                        ),
                                    ),
                                ],
                                className="filters",
                            ),
                            html.Div(
                                [
                                    html.H2("Total demand", id="forecast-title"),
                                    dcc.Graph(
                                        id="forecast-chart",
                                        responsive=True,
                                        style={"height": "270px", "width": "100%"},
                                        figure=empty_figure("Loading saved forecasts."),
                                        config=GRAPH_CONFIG,
                                    ),
                                    html.Div(
                                        "Point forecasts and joint paths add across the hierarchy. "
                                        "Marginal quantiles need not add.",
                                        className="chart-caption",
                                    ),
                                    html.Div(
                                        "Shaded range P10–P95 has nominal coverage of 85%.",
                                        className="chart-caption",
                                    ),
                                ],
                                className="panel forecast-main",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [html.H2("28-day forecast"), html.Div(id="forecast-table")],
                                        className="panel",
                                    ),
                                    html.Div(
                                        [
                                            html.H2("What drives the median model"),
                                            dcc.Graph(
                                                id="shap-chart",
                                                responsive=True,
                                                style={"height": "260px", "width": "100%"},
                                                figure=empty_figure(
                                                    "Loading model explanation.", 260
                                                ),
                                                config=GRAPH_CONFIG,
                                            ),
                                            html.Div(id="shap-note", className="chart-caption"),
                                        ],
                                        className="panel",
                                    ),
                                ],
                                className="forecast-lower",
                            ),
                            html.Div(id="forecast-source", className="source-line"),
                        ],
                        id="forecast-panel",
                    ),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.H1("Replenishment simulator"),
                                    html.P(SIMULATOR_SUBTITLE),
                                ],
                                className="view-heading",
                            ),
                            html.Div(
                                [
                                    html.Aside(
                                        [
                                            field(
                                                "Evaluation window",
                                                dropdown(
                                                    "scenario-phase",
                                                    [
                                                        {"label": v, "value": k}
                                                        for k, v in repository.phase_labels.items()
                                                    ],
                                                    "holdout",
                                                ),
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Demand quantile",
                                                        htmlFor="scenario-quantile",
                                                        className="field-label",
                                                    ),
                                                    dcc.Slider(
                                                        id="scenario-quantile",
                                                        min=0,
                                                        max=5,
                                                        step=1,
                                                        marks={
                                                            i: f"P{round(q * 100):02}"
                                                            for i, q in enumerate(QUANTILE_CHOICES)
                                                        },
                                                        value=4,
                                                        included=False,
                                                    ),
                                                ],
                                                className="slider-field",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        [
                                                            "Store capacity",
                                                            html.Span(
                                                                "100%",
                                                                id="capacity-value",
                                                                className="slider-value",
                                                            ),
                                                        ],
                                                        htmlFor="scenario-capacity",
                                                        className="field-label",
                                                    ),
                                                    dcc.Slider(
                                                        id="scenario-capacity",
                                                        min=10,
                                                        max=150,
                                                        step=5,
                                                        marks={
                                                            v: f"{v}%" for v in [10, 50, 100, 150]
                                                        },
                                                        value=100,
                                                    ),
                                                ],
                                                className="slider-field",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        [
                                                            "Lead-time delay",
                                                            html.Span(
                                                                "+0 days",
                                                                id="delay-value",
                                                                className="slider-value",
                                                            ),
                                                        ],
                                                        htmlFor="scenario-delay",
                                                        className="field-label",
                                                    ),
                                                    dcc.Slider(
                                                        id="scenario-delay",
                                                        min=0,
                                                        max=4,
                                                        step=1,
                                                        marks={v: str(v) for v in range(5)},
                                                        value=0,
                                                    ),
                                                ],
                                                className="slider-field",
                                            ),
                                            html.P(
                                                QUANTILE_NOTE,
                                                id="quantile-note",
                                                className="control-note",
                                            ),
                                            html.Button(
                                                "Run optimizer",
                                                id="run-optimizer",
                                                n_clicks=0,
                                                className="button run-button",
                                            ),
                                            html.Div(
                                                id="submit-error",
                                                role="alert",
                                                className="submission-error",
                                            ),
                                        ],
                                        className="panel scenario-controls",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "Ready",
                                                        id="scenario-status",
                                                        className="status-text",
                                                        role="status",
                                                        **{"aria-live": "polite"},
                                                    ),
                                                    html.Div(
                                                        dropdown(
                                                            "result-scope",
                                                            [
                                                                {
                                                                    "label": "28-day window",
                                                                    "value": "decision_window",
                                                                },
                                                                {
                                                                    "label": TAIL_LABEL,
                                                                    "value": "including_tail",
                                                                },
                                                            ],
                                                            "decision_window",
                                                        ),
                                                        className="scope-field",
                                                    ),
                                                ],
                                                className="status-line",
                                            ),
                                            html.Div(
                                                blank_results(
                                                    "Loading validated reference results."
                                                ),
                                                id="scenario-results",
                                            ),
                                        ],
                                        className="chart-panel",
                                    ),
                                ],
                                className="scenario-layout",
                            ),
                        ],
                        id="scenario-panel",
                        style={"display": "none"},
                    ),
                ],
                className="workspace",
            ),
        ]
    )


def create_app(root=ROOT, repository=None, manager=None):
    root = Path(root).resolve()
    app = Dash(
        __name__,
        title="Supply Chain Engine",
        assets_folder=str(ROOT / "app/assets"),
        suppress_callback_exceptions=True,
        update_title="Updating · Supply Chain Engine",
    )
    try:
        repository = repository or DashboardRepository(root)
        manager = manager or JobManager(
            root, repository.recpath, repository.basepath, config_from_project(root)
        )
    except (ValueError, OSError, KeyError) as error:
        LOGGER.error("Dashboard source unavailable: %s", error)
        app.layout = html.Main(
            [
                html.H1("Forecast data is not ready"),
                html.P(
                    "Complete the ingestion, forecasting, reconciliation and inventory steps, "
                    "then restart this dashboard."
                ),
                html.P("See README.md for pipeline commands."),
            ],
            className="setup panel",
        )
        return app
    app.layout = layout(repository)
    app.repository = repository
    app.job_manager = manager

    @app.callback(
        Output("forecast-panel", "style"),
        Output("scenario-panel", "style"),
        Input("workspace-tab", "value"),
    )
    def switch_tab(tab):
        return (
            ({"display": "block"}, {"display": "none"})
            if tab == "forecast"
            else ({"display": "none"}, {"display": "block"})
        )

    @app.callback(
        Output("category-filter", "options"),
        Output("category-filter", "value"),
        Output("category-filter", "disabled"),
        Input("store-filter", "value"),
    )
    def categories(store):
        opts = [
            {"label": "All categories", "value": ALL},
            *[{"label": c.title(), "value": c} for c in repository.category_options(store)],
        ]
        return opts, ALL, store == ALL

    @app.callback(
        Output("item-filter", "options"),
        Output("item-filter", "value"),
        Output("item-filter", "disabled"),
        Input("store-filter", "value"),
        Input("category-filter", "value"),
    )
    def items(store, category):
        options = (
            repository.item_options(store, category) if store != ALL and category != ALL else []
        )
        return [{"label": "All items", "value": ALL}, *options], ALL, not options

    @app.callback(
        Output("forecast-title", "children"),
        Output("forecast-chart", "figure"),
        Output("forecast-table", "children"),
        Output("shap-chart", "figure"),
        Output("shap-note", "children"),
        Output("forecast-source", "children"),
        Input("forecast-origin", "value"),
        Input("store-filter", "value"),
        Input("category-filter", "value"),
        Input("item-filter", "value"),
    )
    def update_forecast(origin, store, category, item):
        try:
            node, date, observed, forecast = repository.forecast_view(origin, store, category, item)
        except ValueError:
            return (
                "Choose a hierarchy selection",
                empty_figure("Updating the hierarchy selection."),
                [],
                empty_figure(""),
                "",
                "",
            )
        title = (
            "Total demand"
            if node == "Total"
            else f"{store} demand"
            if category == ALL
            else f"{store} · {category.title()}"
            if item == ALL
            else f"{store} · {item.split('/')[-1]}"
        )
        importance, role = repository.shap_importance(origin)
        note = (
            f"Model-wide mean |SHAP| · {'production' if role == 'latest' else 'holdout'} P50. "
            "Associations are not causal effects."
            if role
            else "No explanation sample was saved for this tuning fold."
        )
        source = (
            f"Reconciliation {repository.rec['run_id']} · "
            f"Origin {date:%b %d, %Y} · 28 days · Node {node}"
        )
        return (
            title,
            forecast_figure(date, observed, forecast),
            forecast_table(forecast),
            shap_figure(importance),
            note,
            source,
        )

    @app.callback(
        Output("capacity-value", "children"),
        Output("delay-value", "children"),
        Output("quantile-note", "children"),
        Input("scenario-capacity", "value"),
        Input("scenario-delay", "value"),
        Input("scenario-quantile", "value"),
    )
    def control_labels(capacity, delay, quantile):
        try:
            q = quantile_at(quantile)
            note = f"P{round(q * 100):02} demand is a planning scenario, not a minimum fill rate."
        except ValueError:
            note = "Choose one of the six available forecast quantiles."
        return f"{capacity:g}%", f"+{delay:g} days", note

    def config_for(q, capacity, delay):
        return selected_config(manager.defaults, quantile_at(q), capacity, delay)

    def resolve(state, phase, config):
        key, _ = manager.identity(phase, config)
        if state and state.get("job_id") == key:
            if state["status"] == "completed":
                result = repository.result(str(root / state["result_manifest"]), phase)
                if result["manifest"]["config"] != asdict(config):
                    raise ValueError("Result settings differ from the selected scenario.")
                return result, "Completed"
            return None, state["status"]
        path = repository.reference(phase, asdict(config))
        if path:
            return repository.result(str(path), phase), "Saved reference"
        return None, "ready"

    @app.callback(
        Output("job-id", "data"),
        Output("submit-error", "children"),
        Input("run-optimizer", "n_clicks"),
        State("scenario-phase", "value"),
        State("scenario-quantile", "value"),
        State("scenario-capacity", "value"),
        State("scenario-delay", "value"),
        prevent_initial_call=True,
    )
    def submit(n, phase, q, capacity, delay):
        if not n:
            raise PreventUpdate
        try:
            config = config_for(q, capacity, delay)
            key, _ = manager.identity(phase, config)
            if manager.path(key).exists():
                manager.read(key)
            reference = repository.reference(phase, asdict(config))
            if reference:
                repository.result(str(reference), phase)
            state = manager.submit(phase, config, reference=reference)
            return {"job_id": state["job_id"]}, ""
        except (ValueError, OSError) as error:
            LOGGER.error("Could not submit scenario: %s", error)
            return no_update, "The scenario could not start. Check the selected settings and retry."

    @app.callback(
        Output("job-state", "data"),
        Input("job-id", "data"),
        Input("job-poll", "n_intervals"),
        State("job-state", "data"),
    )
    def poll(job, _, previous):
        if not job:
            return no_update
        try:
            state = manager.read(job["job_id"])
        except (ValueError, OSError, KeyError):
            state = {
                "job_id": job.get("job_id"),
                "status": "failed",
                "error": "This scenario job is unavailable. Run optimizer to retry.",
            }
        return no_update if state == previous else state

    @app.callback(
        Output("scenario-results", "children"),
        Output("scenario-status", "children"),
        Output("scenario-status", "className"),
        Output("run-optimizer", "disabled"),
        Output("run-optimizer", "children"),
        *[
            Output(c, "disabled")
            for c in ["scenario-phase", "scenario-quantile", "scenario-capacity", "scenario-delay"]
        ],
        Input("job-state", "data"),
        Input("scenario-phase", "value"),
        Input("scenario-quantile", "value"),
        Input("scenario-capacity", "value"),
        Input("scenario-delay", "value"),
        Input("result-scope", "value"),
    )
    def show_results(state, phase, q, capacity, delay, scope):
        try:
            if scope not in {"decision_window", "including_tail"}:
                raise ValueError("Unknown result window.")
            config = config_for(q, capacity, delay)
            result, status = resolve(state, phase, config)
            active = status in ACTIVE
            label = (
                "Queued"
                if status == "pending"
                else "Optimizer running…"
                if status == "running"
                else "Run optimizer"
            )
            if result is not None:
                content = scenario_results(result, phase, scope)
                status_text = (
                    f"{status} · P{round(config.service_quantile * 100):02} · "
                    f"{capacity:g}% capacity · +{delay:g} days"
                )
            elif status == "failed":
                content = blank_results(
                    state.get("error", "The scenario failed. Retry with different settings."), True
                )
                status_text = "Failed · no orders available"
            elif active:
                text = (
                    "Waiting for the available solver worker."
                    if status == "pending"
                    else "Optimization is running in the background. Solver budget is 60 seconds; "
                    "data preparation and replay add time."
                )
                content = blank_results(text)
                status_text = (
                    "Pending · queued for the solver"
                    if status == "pending"
                    else "Running · scenario controls are locked"
                )
            else:
                content = blank_results(
                    "Settings changed. Run optimizer to calculate this scenario."
                )
                status_text = "Ready · no result for these settings"
            return content, status_text, f"status-text {status}", active, label, *([active] * 4)
        except (ValueError, OSError, KeyError) as error:
            LOGGER.error("Could not display scenario: %s", error)
            return (
                blank_results(
                    "The saved result could not be verified. Run a new scenario or restart "
                    "after checking the source files.",
                    True,
                ),
                "Result unavailable",
                "status-text",
                False,
                "Run optimizer",
                False,
                False,
                False,
                False,
            )

    @app.callback(
        Output("order-download", "data"),
        Input("download-orders", "n_clicks"),
        State("job-state", "data"),
        State("scenario-phase", "value"),
        State("scenario-quantile", "value"),
        State("scenario-capacity", "value"),
        State("scenario-delay", "value"),
        State("result-scope", "value"),
        prevent_initial_call=True,
    )
    def download(n, state, phase, q, capacity, delay, scope):
        if not n:
            raise PreventUpdate
        try:
            result, _ = resolve(state, phase, config_for(q, capacity, delay))
            if result is None or not result_view(result, phase, scope)["primary"].capacity_feasible:
                raise PreventUpdate
            name = f"orders_{phase}_{result['manifest']['run_id']}.csv"
            return dcc.send_data_frame(result["orders"].to_csv, name, index=False)
        except (ValueError, OSError, KeyError):
            raise PreventUpdate from None

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    load_dotenv(args.root / ".env", override=False)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    host = args.host or os.environ.get("DASH_HOST", "127.0.0.1")
    port = args.port or int(os.environ.get("DASH_PORT", "8050"))
    create_app(args.root).run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
