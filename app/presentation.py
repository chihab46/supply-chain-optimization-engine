"""Code-native charts and compact result views for the planning workbench."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html

from supply_chain_opt_engine.dashboard_data import POLICY_LABELS
from supply_chain_opt_engine.optimization import MILP

INK = "#20332e"
GREEN = "#237558"
MUTED = "#66756d"


def theme(fig, height=340):
    fig.update_layout(
        height=height,
        margin=dict(l=54, r=22, t=20, b=48),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Lato, sans-serif", size=13, color=INK),
        hovermode="x unified",
        legend=dict(
            orientation="h", y=1.1, x=1.0, xanchor="right", traceorder="normal", font=dict(size=12)
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor="rgba(32,51,46,.08)", zeroline=False, automargin=True)
    return fig


def empty_figure(message, height=320):
    fig = theme(go.Figure(), height)
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color=MUTED, size=14),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def forecast_figure(origin, observed, forecast):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast.ds, y=forecast.q10, line=dict(width=0), showlegend=False, hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.q95,
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(80,170,110,.12)",
            name="P10–P95",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.ds, y=forecast.q10, line=dict(width=0), showlegend=False, hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.q90,
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(80,170,110,.16)",
            name="P10–P90",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(x=observed.ds, y=observed.y, name="Actual", line=dict(color=INK, width=1.7))
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.point,
            name="Coherent center",
            line=dict(color=GREEN, width=2.5),
            customdata=forecast[["q50"]].to_numpy(),
            hovertemplate="Center %{y:,.2f}<br>Marginal P50 %{customdata[0]:,.2f}<extra></extra>",
        )
    )
    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=origin,
        x1=origin,
        y0=0,
        y1=1,
        line=dict(color=MUTED, width=1, dash="dot"),
    )
    theme(fig, 270)
    limits = np.concatenate(
        [observed.y.to_numpy(), forecast[["q10", "q95", "point"]].to_numpy().ravel()]
    )
    lower = max(0.0, float(limits.min()) * 0.95)
    upper = max(lower + 1.0, float(limits.max()) * 1.05)
    fig.update_yaxes(title="Daily demand · units", range=[lower, upper])
    return fig


def shap_figure(frame):
    if frame is None:
        return empty_figure("No SHAP sample saved for this tuning fold.", 260)
    names = {
        "rolling_mean_7": "Recent 7-day mean",
        "rolling_mean_14": "Recent 14-day mean",
        "rolling_mean_28": "Recent 28-day mean",
        "series_id": "Series identity",
        "horizon": "Days ahead",
        "lag_7": "7-day lag",
        "price_relative_to_52w": "Relative price",
        "snap_CA": "SNAP calendar",
        "horizon_day": "Days ahead",
    }
    data = frame.iloc[::-1]
    fig = theme(
        go.Figure(
            go.Bar(
                x=data.mean_absolute_shap,
                y=[names.get(s, s.replace("_", " ")) for s in data.feature],
                orientation="h",
                marker_color=GREEN,
                hovertemplate="%{y}: %{x:.3f} units<extra></extra>",
            )
        ),
        260,
    )
    fig.update_layout(margin=dict(l=140, r=20, t=10, b=40), hovermode="closest")
    fig.update_xaxes(title="Mean |SHAP| · units", showgrid=True, gridcolor="rgba(32,51,46,.08)")
    fig.update_yaxes(showgrid=False)
    return fig


def table(headers, rows, *, caption=None):
    return html.Div(
        html.Table(
            [
                *([html.Caption(caption)] if caption else []),
                html.Thead(html.Tr([html.Th(h, scope="col") for h in headers])),
                html.Tbody([html.Tr([html.Td(v) for v in row]) for row in rows]),
            ]
        ),
        className="table-scroll",
    )


def forecast_table(frame):
    return table(
        ["Date", "Center", "P10", "P50", "P90", "P95"],
        [
            [
                r.ds.strftime("%b %d, %Y"),
                *[f"{getattr(r, c):,.2f}" for c in ["point", "q10", "q50", "q90", "q95"]],
            ]
            for r in frame.itertuples()
        ],
        caption="28 daily forecast observations",
    )


def inventory_figure(trace):
    grouped = trace.groupby("ds", as_index=False)[["inventory", "arrivals"]].sum()
    fig = go.Figure(
        go.Bar(x=grouped.ds, y=grouped.arrivals, name="Receipts", marker_color="#cbe8c5")
    )
    fig.add_trace(
        go.Scatter(
            x=grouped.ds,
            y=grouped.inventory,
            name="Ending inventory",
            line=dict(color=GREEN, width=2.3),
        )
    )
    theme(fig, 290)
    fig.update_yaxes(title="Units", rangemode="tozero")
    return fig


def money(value):
    return "—" if value is None or not np.isfinite(value) else f"${value:,.2f}"


def rate(value):
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.2%}"


def result_view(result, phase, scope):
    evaluation = "realized" if phase == "holdout" else "forecast"
    metrics = result["metrics"]
    selected = metrics[metrics.evaluation.eq(evaluation) & metrics.scope.eq(scope)]
    primary = selected[selected.policy.eq(MILP)].iloc[0]
    days = int(primary.days)
    trace = result["trace"]
    trace = trace[trace.evaluation.eq(evaluation) & trace.policy.eq(MILP)]
    dates = sorted(trace.ds.unique())[:days]
    trace = trace[trace.ds.isin(dates)]
    columns = [
        "Policy",
        "Total cost",
        "Fill rate" if phase == "holdout" else "Projected fulfillment",
        "Storage",
    ]
    comparison = table(
        columns,
        [
            [
                POLICY_LABELS.get(r.policy, r.policy),
                money(r.total_cost),
                rate(r.fill_rate),
                "Within limit" if r.capacity_feasible else "Exceeds capacity",
            ]
            for r in selected.itertuples()
        ],
    )
    breakdown = table(
        ["Cost component", "USD"],
        [
            ["Holding", money(primary.holding_cost)],
            ["Stockouts", money(primary.stockout_cost)],
            ["Ordering fees", money(primary.fixed_order_cost)],
        ],
    )
    solver = table(
        ["Measure", "Result"],
        [
            ["Certified gap", rate(result["gap"])],
            ["Capacity", "Within limit" if primary.capacity_feasible else "Exceeds capacity"],
            ["Decision horizon", "28 days"],
            ["Tail", f"{result['manifest']['config']['tail_days']} days"],
        ],
    )
    warning = []
    if primary.order_count == 0:
        warning.append(
            "No new orders selected. Current costs favor shortages; "
            "no minimum service constraint is applied."
        )
    if not primary.capacity_feasible:
        warning.append(
            "This realized replay exceeds storage. Download is disabled for this window."
        )
    if not selected.capacity_feasible.all():
        warning.append(
            "A comparison policy exceeds storage in this window. "
            "Its scores are diagnostic, not a feasible operating plan."
        )
    if phase == "latest":
        warning.append(
            "Forecast-demand replay. Realized future demand and fill rate are unavailable."
        )
    return {
        "primary": primary,
        "figure": inventory_figure(trace),
        "comparison": comparison,
        "breakdown": breakdown,
        "solver": solver,
        "warning": " ".join(warning),
        "fill_label": "Realized fill rate" if phase == "holdout" else "Projected fulfillment",
        "caption": "Holdout realized demand"
        if phase == "holdout"
        else "Latest forecast-demand scenario",
    }


def kpi(label, value):
    return html.Div(
        [html.Div(label, className="metric-label"), html.Div(value, className="metric-value")],
        className="metric",
    )


def blank_results(message, failed=False):
    return html.Div(
        [
            html.Div(
                [
                    kpi("Total cost", "—"),
                    kpi("Fill rate", "—"),
                    kpi("Avg inventory value", "—"),
                    html.Div(
                        [
                            html.Button(
                                "Download orders",
                                id="download-orders",
                                disabled=True,
                                className="button",
                            ),
                            html.Div("No current result", className="muted"),
                        ],
                        className="metric order-metric",
                    ),
                ],
                className="metrics-band panel",
            ),
            html.Div(message, className="notice error" if failed else "notice", role="status"),
            html.Div(
                [
                    html.H2("Inventory and receipts"),
                    dcc.Graph(
                        figure=empty_figure("Run a scenario to calculate the inventory plan."),
                        responsive=True,
                        style={"height": "320px", "width": "100%"},
                        config={"displayModeBar": False, "responsive": True},
                    ),
                ],
                className="panel",
            ),
        ],
        className="result-stack",
    )


def scenario_results(result, phase, scope):
    view = result_view(result, phase, scope)
    r = view["primary"]
    return html.Div(
        [
            html.Div(
                [
                    kpi("Total cost", money(r.total_cost)),
                    kpi(view["fill_label"], rate(r.fill_rate)),
                    kpi("Avg inventory value", money(r.average_inventory_value)),
                    html.Div(
                        [
                            html.Button(
                                "Download orders",
                                id="download-orders",
                                disabled=not bool(r.capacity_feasible),
                                className="button",
                            ),
                            html.Div(f"{int(r.order_count):,} orders", className="muted"),
                        ],
                        className="metric order-metric",
                    ),
                ],
                className="metrics-band panel",
            ),
            *(
                [html.Div(view["warning"], className="notice warning", role="status")]
                if view["warning"]
                else []
            ),
            html.Div(
                [
                    html.H2("Inventory and receipts"),
                    dcc.Graph(
                        figure=view["figure"],
                        responsive=True,
                        style={"height": "290px", "width": "100%"},
                        config={"displayModeBar": False, "responsive": True},
                    ),
                    html.Div(view["caption"], className="chart-caption"),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div([html.H2("Cost breakdown"), view["breakdown"]], className="panel"),
                    html.Div([html.H2("Solver summary"), view["solver"]], className="panel"),
                ],
                className="result-pair",
            ),
            html.Div([html.H2("Policy comparison"), view["comparison"]], className="panel"),
            html.Div(
                f"Result {result['manifest']['run_id']} · "
                "Same inventory, receipts and economics across policies.",
                className="source-line",
            ),
        ],
        className="result-stack",
    )
