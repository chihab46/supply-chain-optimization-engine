from __future__ import annotations

import json
import os
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("dash")

from app.dashboard import create_app
from app.presentation import result_view

from supply_chain_opt_engine.dashboard_data import ALL, DashboardRepository
from supply_chain_opt_engine.dashboard_jobs import (
    JobManager,
    atomic_json,
    execute_job,
    selected_config,
)
from supply_chain_opt_engine.hierarchy import Hierarchy
from supply_chain_opt_engine.inventory_data import InventoryInputs
from supply_chain_opt_engine.m5 import checksum
from supply_chain_opt_engine.optimization import MEDIAN, MILP, OptimizationConfig
from supply_chain_opt_engine.simulation import evaluate_policy, purchase_orders


@pytest.fixture
def repository(tmp_path):
    repo = object.__new__(DashboardRepository)
    repo.root = tmp_path
    repo.recpath = tmp_path / "rec.json"
    repo.recpath.write_text("{}")
    repo.basepath = tmp_path / "base.json"
    repo.basepath.write_text("{}")
    repo.rec = {
        "run_id": "rec-unit",
        "config": {"primary_method": "mint_shrink"},
        "latest_origin": "2020-03-01",
    }
    repo.holdout = {"fold_id": 4, "origin": "2020-01-01", "role": "holdout"}
    catalog = pd.DataFrame(
        {
            "unique_id": ["S/a", "S/b"],
            "store_id": ["S", "S"],
            "cat_id": ["FOODS", "HOUSEHOLD"],
            "item_id": ["a", "b"],
        }
    )
    repo.hierarchy = Hierarchy.from_catalog(catalog)
    repo.nodes = repo.hierarchy.nodes
    repo.leaves = repo.nodes[repo.nodes.level.eq("store_item")]
    repo.node_indices = dict(zip(repo.nodes.unique_id, repo.nodes.index, strict=True))
    repo.history_dates = pd.date_range("2019-09-01", periods=210)
    repo.history = repo.hierarchy.aggregate(np.ones((2, 210)))
    repo.origins = {
        "latest": pd.Timestamp("2020-03-01"),
        "fold_4": pd.Timestamp("2020-01-01"),
        "fold_1": pd.Timestamp("2019-12-01"),
    }
    repo.origin_labels = {k: k for k in repo.origins}
    repo.phase_labels = {"holdout": "Holdout", "latest": "Latest"}
    repo.forecasts = {}
    for key, origin in repo.origins.items():
        dates = pd.date_range(origin, periods=29)[1:]
        frame = pd.DataFrame(
            {
                "unique_id": np.repeat(repo.nodes.unique_id, 28),
                "ds": np.tile(dates, len(repo.nodes)),
                "point": 1.0,
            }
        )
        for q in ["q05", "q10", "q50", "q80", "q90", "q95"]:
            frame[q] = 1.0
        repo.forecasts[key] = frame
    repo.globalpath = tmp_path / "global/manifest.json"
    repo.globalpath.parent.mkdir()
    shap = repo.globalpath.parent / "latest_shap_importance.csv"
    pd.DataFrame({"feature": ["lag_7"], "mean_absolute_shap": [1.0], "sample_rows": [2]}).to_csv(
        shap, index=False
    )
    hold = repo.globalpath.parent / "holdout_shap_importance.csv"
    hold.write_bytes(shap.read_bytes())
    repo.global_manifest = {"artifact_sha256": {p.name: checksum(p) for p in [shap, hold]}}
    repo.globalpath.write_text(json.dumps(repo.global_manifest))
    bundle = tmp_path / "artifacts/orders/optimization/unit"
    bundle.mkdir(parents=True)
    config = OptimizationConfig()
    metrics = []
    c = catalog.assign(
        lead_time_days=1,
        minimum_order_quantity=1,
        unit_volume_m3=0.01,
        daily_holding_cost=0.1,
        stockout_penalty=2.0,
        fixed_order_cost=3.0,
        unit_purchase_cost=10.0,
    )
    phases = []
    for phase, key in [("holdout", "fold_4"), ("latest", "latest")]:
        d = bundle / phase
        d.mkdir()
        origin = repo.origins[key]
        data = InventoryInputs(
            c,
            pd.date_range(origin, periods=43)[1:],
            origin,
            28,
            np.ones((2, 42)),
            np.array([3, 3]),
            np.zeros((2, 42), dtype=int),
            {"S": 100.0},
        )
        orders = np.zeros((2, 28), dtype=int)
        tables = []
        for policy in [MILP, MEDIAN]:
            for evaluation in ["forecast", "realized"] if phase == "holdout" else ["forecast"]:
                scores, trace = evaluate_policy(data, orders, policy=policy, evaluation=evaluation)
                metrics.extend([{**r, "phase": phase} for r in scores])
                tables.append(trace)
        pd.concat(tables).to_parquet(d / "inventory_trace.parquet", index=False)
        purchase_orders(data, orders, policy=MILP, scenario_id=phase).to_csv(
            d / "purchase_orders.csv", index=False
        )
        (d / "solver_diagnostics.json").write_text(
            json.dumps(
                [
                    {
                        "has_feasible_solution": True,
                        "incumbent_objective": 100.0,
                        "objective_bound": 99.0,
                    }
                ]
            )
        )
        phases.append({"phase": phase})
    pd.DataFrame(metrics).to_csv(bundle / "policy_metrics.csv", index=False)
    manifest = {
        "status": "optimized",
        "run_id": "unit",
        "config": asdict(config),
        "phases": phases,
        "reconciliation_run_id": "rec-unit",
        "reconciliation_manifest_sha256": checksum(repo.recpath),
        "artifact_sha256": {
            str(p.relative_to(bundle)): checksum(p) for p in bundle.rglob("*") if p.is_file()
        },
    }
    path = bundle / "manifest.json"
    path.write_text(json.dumps(manifest))
    repo.reference_runs = [(path, manifest)]
    return repo


def call(app, target, values, states=None, changed=None):
    key = next(k for k in app.callback_map if target in k)
    entry = app.callback_map[key]
    output = entry["output"]
    payload = {
        "output": key,
        "outputs": [o.to_dict() for o in output] if isinstance(output, list) else output.to_dict(),
        "inputs": [{**i, "value": values[i["id"]]} for i in entry["inputs"]],
        "state": [{**s, "value": (states or {})[s["id"]]} for s in entry["state"]],
        "changedPropIds": [
            changed or f"{entry['inputs'][0]['id']}.{entry['inputs'][0]['property']}"
        ],
    }
    return app.server.test_client().post("/_dash-update-component", json=payload)


@pytest.fixture
def ui(repository):
    launched = []

    def launcher(*args, **kwargs):
        launched.append(args[0])
        return SimpleNamespace(pid=os.getpid())

    manager = JobManager(
        repository.root, repository.recpath, repository.basepath, launcher=launcher
    )
    return create_app(repository.root, repository, manager), manager, launched


def test_app_serves_local_assets_and_chained_hierarchy(ui):
    app, _, _ = ui
    client = app.server.test_client()
    for path in [
        "/",
        "/_dash-layout",
        "/_dash-dependencies",
        "/assets/dashboard.css",
        "/assets/fonts/Lato-Regular.ttf",
    ]:
        assert client.get(path).status_code == 200
    result = call(app, "category-filter.options", {"store-filter": "S"}).json["response"]
    assert {o["value"] for o in result["category-filter"]["options"]} == {ALL, "FOODS", "HOUSEHOLD"}
    result = call(
        app, "item-filter.options", {"store-filter": "S", "category-filter": "FOODS"}
    ).json["response"]
    assert result["item-filter"]["options"][-1]["value"] == "S/a"
    result = call(
        app,
        "forecast-title.children",
        {
            "forecast-origin": "latest",
            "store-filter": "S",
            "category-filter": "FOODS",
            "item-filter": "S/a",
        },
    ).json["response"]
    assert "S" in result["forecast-title"]["children"]
    assert len(result["forecast-chart"]["figure"]["data"][-1]["x"]) == 28
    assert "production P50" in result["shap-note"]["children"]
    response = call(
        app,
        "forecast-title.children",
        {
            "forecast-origin": "fold_1",
            "store-filter": ALL,
            "category-filter": ALL,
            "item-filter": ALL,
        },
    ).json["response"]
    assert "No explanation sample" in response["shap-note"]["children"]


def test_hierarchy_rejects_mismatched_and_unknown_nodes(repository):
    assert repository.node_id() == "Total"
    assert repository.node_id("S") == "Store/S"
    assert repository.node_id("S", "FOODS") == "StoreCategory/S/FOODS"
    with pytest.raises(ValueError):
        repository.node_id("S", "FOODS", "S/b")
    with pytest.raises(ValueError):
        repository.node_id(ALL, "FOODS")
    with pytest.raises(ValueError):
        repository.forecast_view("unknown")


def test_scenario_origin_labels_no_stale_results_and_empty_download(ui):
    app, _, _ = ui
    values = {
        "job-state": None,
        "scenario-phase": "holdout",
        "scenario-quantile": 4,
        "scenario-capacity": 100,
        "scenario-delay": 0,
        "result-scope": "decision_window",
    }
    response = call(app, "scenario-results.children", values).json["response"]
    rendered = json.dumps(response)
    assert "Realized fill rate" in rendered and "No new orders selected" in rendered
    assert "Saved reference" in response["scenario-status"]["children"]
    values["scenario-phase"] = "latest"
    rendered = json.dumps(call(app, "scenario-results.children", values).json)
    assert "Projected fulfillment" in rendered and "Realized fill rate" not in rendered
    values["scenario-capacity"] = 75
    response = call(app, "scenario-results.children", values).json["response"]
    assert response["scenario-status"]["children"] == "Ready · no result for these settings"
    assert "No current result" in json.dumps(response)
    state = {
        "job-state": None,
        "scenario-phase": "holdout",
        "scenario-quantile": 4,
        "scenario-capacity": 100,
        "scenario-delay": 0,
        "result-scope": "decision_window",
    }
    download = call(app, "order-download.data", {"download-orders": 1}, state)
    assert download.status_code == 200
    csv = download.json["response"]["order-download"]["data"]
    assert "quantity" in csv["content"] and len(csv["content"].strip().splitlines()) == 1
    state["scenario-capacity"] = 75
    assert call(app, "order-download.data", {"download-orders": 1}, state).status_code == 204


def test_background_submission_deduplicates_and_locks_controls(ui):
    app, manager, launched = ui
    states = {
        "scenario-phase": "holdout",
        "scenario-quantile": 3,
        "scenario-capacity": 100,
        "scenario-delay": 0,
    }
    a = call(app, "job-id.data", {"run-optimizer": 1}, states).json["response"]["job-id"]["data"]
    b = call(app, "job-id.data", {"run-optimizer": 2}, states).json["response"]["job-id"]["data"]
    assert a == b and len(launched) == 1
    pending = manager.read(a["job_id"])
    response = call(
        app,
        "scenario-results.children",
        {"job-state": pending, **states, "result-scope": "decision_window"},
    ).json["response"]
    assert response["run-optimizer"]["disabled"]
    for c in ["scenario-phase", "scenario-quantile", "scenario-capacity", "scenario-delay"]:
        assert response[c]["disabled"]
    assert "Pending" in response["scenario-status"]["children"]
    pending["status"] = "failed"
    pending["error"] = "No feasible plan."
    response = call(
        app,
        "scenario-results.children",
        {"job-state": pending, **states, "result-scope": "decision_window"},
    ).json["response"]
    assert not response["run-optimizer"]["disabled"]
    assert "No feasible plan." in json.dumps(response)


def test_worker_completes_or_fails_without_fabricating_orders(ui, repository):
    _, manager, launched = ui
    state = manager.submit("latest", OptimizationConfig())
    path = repository.reference_runs[0][0]
    result = execute_job(repository.root, state["job_id"], runner=lambda *a, **k: path.parent)
    assert result["status"] == "completed" and result["result_manifest"].endswith("manifest.json")
    assert manager.submit("latest", OptimizationConfig())["status"] == "completed"
    assert len(launched) == 1
    bad = manager.submit("holdout", selected_config(OptimizationConfig(), 0.05, 10, 4))

    def fail(*args, **kwargs):
        raise ValueError("No complete feasible plan. Inspect failure manifest.")

    failed = execute_job(repository.root, bad["job_id"], runner=fail)
    assert failed["status"] == "failed" and failed["result_manifest"] is None
    assert "No feasible plan" in failed["error"]
    assert (
        manager.submit("holdout", selected_config(OptimizationConfig(), 0.05, 10, 4))["status"]
        == "pending"
    )


def test_job_identity_and_stale_worker_detection(ui):
    _, manager, _ = ui
    config = OptimizationConfig()
    key, _ = manager.identity("holdout", config)
    assert key == manager.identity("holdout", config)[0]
    assert key != manager.identity("latest", config)[0]
    state = manager.submit("holdout", config)
    state.update(worker_pid=99999999, updated_at="2000-01-01T00:00:00+00:00")
    atomic_json(manager.path(key), state)
    assert manager.read(key)["status"] == "failed"
    with pytest.raises(ValueError):
        manager.path("../escape")
    with pytest.raises(ValueError):
        selected_config(config, 0.9, 5, 0)
    with pytest.raises(ValueError):
        selected_config(config, 0.9, 100, 1.5)


def test_result_verifies_hashes_and_restricts_file_paths(repository):
    path = repository.reference_runs[0][0]
    result = repository.result(str(path), "holdout")
    view = result_view(result, "holdout", "decision_window")
    assert view["fill_label"] == "Realized fill rate"
    assert view["primary"].order_count == 0
    assert result_view(result, "latest", "including_tail")["fill_label"] == "Projected fulfillment"
    with pytest.raises(ValueError):
        repository.result(str(repository.root / "rec.json"), "holdout")
    (path.parent / "holdout/purchase_orders.csv").write_text("tampered")
    with pytest.raises(ValueError, match="verification"):
        repository.result(str(path), "holdout")


def test_missing_data_has_actionable_empty_state(tmp_path):
    app = create_app(tmp_path)
    response = app.server.test_client().get("/_dash-layout")
    assert response.status_code == 200 and b"Forecast data is not ready" in response.data


def test_slider_whole_number_floats_are_validated_and_converted(ui):
    from app.dashboard import quantile_at

    assert quantile_at(3.0) == 0.8
    assert quantile_at(4.0) == 0.9
    for bad in [True, 0.5, None, float("nan"), 6.0]:
        with pytest.raises(ValueError):
            quantile_at(bad)
    app, _, _ = ui
    labels = call(
        app,
        "capacity-value.children",
        {"scenario-capacity": 100.0, "scenario-delay": 0.0, "scenario-quantile": 3.0},
    )
    assert labels.status_code == 200
    assert "P80" in labels.json["response"]["quantile-note"]["children"]
    response = call(
        app,
        "scenario-results.children",
        {
            "job-state": None,
            "scenario-phase": "holdout",
            "scenario-quantile": 3.0,
            "scenario-capacity": 100.0,
            "scenario-delay": 0.0,
            "result-scope": "decision_window",
        },
    )
    assert response.status_code == 200
    assert "Ready" in response.json["response"]["scenario-status"]["children"]


def test_quantile_slider_has_explicit_integer_step_and_rejects_between_marks(ui):
    app, _, _ = ui
    document = app.server.test_client().get("/_dash-layout").json

    def components(value):
        if isinstance(value, dict):
            yield value
            for item in value.values():
                yield from components(item)
        elif isinstance(value, list):
            for item in value:
                yield from components(item)

    slider = next(
        c for c in components(document) if c.get("props", {}).get("id") == "scenario-quantile"
    )
    assert slider["props"]["step"] == 1
    labels = call(
        app,
        "capacity-value.children",
        {"scenario-capacity": 100.0, "scenario-delay": 0.0, "scenario-quantile": 0.15},
    )
    assert labels.status_code == 200
    assert "six available" in labels.json["response"]["quantile-note"]["children"]


def test_time_limited_result_without_lower_bound_has_undefined_gap(repository):
    path = repository.reference_runs[0][0]
    manifest = json.loads(path.read_text())
    solver = path.parent / "holdout/solver_diagnostics.json"
    solver.write_text(
        json.dumps(
            [{"has_feasible_solution": True, "incumbent_objective": 100.0, "objective_bound": None}]
        )
    )
    manifest["artifact_sha256"]["holdout/solver_diagnostics.json"] = checksum(solver)
    path.write_text(json.dumps(manifest))
    result = repository.result(str(path), "holdout")
    assert result["gap"] is None
    assert "N/A" in json.dumps(
        result_view(result, "holdout", "decision_window")["solver"].to_plotly_json(), default=str
    )
