from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pyomo.contrib.appsi.base import TerminationCondition as Term
from pyomo.contrib.appsi.solvers import Highs

from supply_chain_opt_engine.inventory_data import (
    InventoryInputs,
    forecast_matrix,
    generate_scenario,
)
from supply_chain_opt_engine.inventory_model import (
    order_bounds,
    solution_acceptable,
    solve_inventory,
    validate_solution,
)
from supply_chain_opt_engine.optimization import (
    MEDIAN,
    MILP,
    OptimizationConfig,
    PlanningFailure,
    run_phase,
)
from supply_chain_opt_engine.simulation import (
    evaluate_policy,
    inventory_trace,
    order_up_to,
    purchase_orders,
)


def inputs(
    demand=((0.0, 5.0, 5.0),),
    initial=None,
    receipts=None,
    *,
    decision_days=None,
    moq=6,
    lead=1,
    capacity=100,
    holding=0.1,
    penalty=5,
    fee=2,
):
    d = np.array(demand, dtype=float)
    n, h = d.shape
    c = pd.DataFrame(
        {
            "unique_id": [f"S/i{i}" for i in range(n)],
            "store_id": ["S"] * n,
            "item_id": [f"i{i}" for i in range(n)],
            "lead_time_days": [lead] * n,
            "minimum_order_quantity": [moq] * n,
            "unit_volume_m3": [1.0] * n,
            "daily_holding_cost": [holding] * n,
            "stockout_penalty": [penalty] * n,
            "fixed_order_cost": [fee] * n,
            "unit_purchase_cost": [10.0] * n,
            "recent_std": [1.0] * n,
        }
    )
    origin = pd.Timestamp("2020-01-01")
    return InventoryInputs(
        c,
        pd.date_range(origin + pd.Timedelta(days=1), periods=h),
        origin,
        h if decision_days is None else decision_days,
        d,
        np.zeros(n, dtype=int) if initial is None else np.array(initial),
        np.zeros((n, h), dtype=int) if receipts is None else np.array(receipts),
        {"S": float(capacity)},
    )


def test_tiny_milp_matches_exhaustive_enumeration_and_costs():
    data = inputs()
    solution = solve_inventory(data, time_limit=5, mip_gap=0)
    assert solution.diagnostics["has_feasible_solution"]
    assert solution.diagnostics["relative_gap"] == pytest.approx(0)
    values = []
    for a in [0, *range(6, 11)]:
        for b in [0, 6]:
            orders = np.array([[a, b, 0]])
            score, _ = evaluate_policy(data, orders, policy="x", evaluation="forecast")
            values.append(score[0]["total_cost"])
    assert solution.diagnostics["incumbent_objective"] == pytest.approx(min(values))
    assert solution.diagnostics["incumbent_objective"] == pytest.approx(2.5)
    np.testing.assert_array_equal(solution.orders, [[10, 0, 0]])
    np.testing.assert_allclose(solution.inventory, [[0, 5, 0]])
    np.testing.assert_allclose(solution.unmet, 0)


def test_fractional_forecast_integer_orders_and_no_rationing():
    data = inputs(((0, 1.5, 1.5),), moq=1, fee=0, holding=1, penalty=100)
    solution = solve_inventory(data, time_limit=5, mip_gap=0)
    np.testing.assert_array_equal(solution.orders, [[2, 1, 0]])
    assert np.max(solution.inventory * solution.unmet) < 1e-8
    assert solution.diagnostics["incumbent_objective"] == pytest.approx(0.5)
    # Flow balance alone could pass I=U=1; physical replay must reject this.
    bad = inputs(((1.0,),), initial=[1], moq=1)
    with pytest.raises(ValueError, match="work-conserving"):
        validate_solution(bad, np.zeros((1, 1)), np.ones((1, 1)), np.ones((1, 1)))


def test_shared_capacity_moq_and_pipeline_infeasibility():
    data = inputs(((0, 2, 2), (0, 2, 2)), moq=6, capacity=3, fee=0, holding=0.1, penalty=100)
    solution = solve_inventory(data, time_limit=5, mip_gap=0)
    assert solution.diagnostics["has_feasible_solution"]
    assert solution.inventory.sum(axis=0).max() <= 3 + 1e-6
    # No legal MOQ fits the remaining demand plus storage.
    np.testing.assert_array_equal(solution.orders, 0)
    infeasible = inputs(((0.0, 0.0),), initial=[4], capacity=3)
    failed = solve_inventory(infeasible, time_limit=5)
    assert failed.orders is None and failed.diagnostics["termination"] == "infeasible"
    pipeline = inputs(((0.0, 0.0),), receipts=[[0, 4]], capacity=3)
    assert solve_inventory(pipeline, time_limit=5).orders is None


def test_lead_time_arrivals_tail_and_order_bounds():
    data = inputs(
        ((3, 3, 3, 3, 3),),
        initial=[1],
        receipts=[[0, 2, 0, 0, 0]],
        decision_days=3,
        moq=1,
        lead=2,
        fee=0,
    )
    trace = inventory_trace(data, np.array([[4, 0, 1]]))
    np.testing.assert_array_equal(trace["arrivals"], [[0, 2, 4, 0, 1]])
    np.testing.assert_array_equal(trace["unmet"], [[2, 1, 0, 2, 2]])
    late = replace(
        data, dates=data.dates[:3], demand=data.demand[:, :3], receipts=data.receipts[:, :3]
    )
    with pytest.raises(ValueError, match="beyond"):
        inventory_trace(late, np.array([[0, 1, 0]]))
    assert order_bounds(late)[0, -1] == 0
    with pytest.raises(ValueError, match="MOQ"):
        inventory_trace(inputs(), np.array([[1, 0, 0]]))


def test_economically_dominated_orders_and_zero_demand():
    data = inputs(((0, 1, 1),), fee=100, penalty=1)
    assert not order_bounds(data).any()
    solution = solve_inventory(data, time_limit=5)
    assert solution.orders.sum() == 0
    empty = inputs(((0.0, 0.0, 0.0),))
    solution = solve_inventory(empty, time_limit=5)
    assert solution.diagnostics["has_feasible_solution"] and solution.orders.sum() == 0
    score, _ = evaluate_policy(empty, solution.orders, policy="zero", evaluation="forecast")
    assert score[0]["fill_rate"] is None and score[0]["total_cost"] == 0


def test_status_gate_and_time_limited_incumbent():
    assert not solution_acceptable(Term.maxTimeLimit, None)
    assert not solution_acceptable(Term.infeasible, 1.0)
    assert not solution_acceptable(Term.error, 1.0)
    assert solution_acceptable(Term.maxTimeLimit, 1.0)

    class NoSolution:
        config = SimpleNamespace()
        highs_options = {}

        def solve(self, model):
            return SimpleNamespace(
                termination_condition=Term.maxTimeLimit,
                best_feasible_objective=None,
                best_objective_bound=0.0,
            )

    failed = solve_inventory(inputs(), solver=NoSolution())
    assert failed.orders is None

    class LimitedHighs(Highs):
        def solve(self, model):
            result = super().solve(model)
            result.termination_condition = Term.maxTimeLimit
            result.best_objective_bound = result.best_feasible_objective * 0.9
            return result

    good = solve_inventory(inputs(), time_limit=5, solver=LimitedHighs())
    assert good.orders is not None and good.diagnostics["relative_gap"] == pytest.approx(0.1)
    assert good.diagnostics["termination"] == "maxTimeLimit"
    assert not good.diagnostics["proven_optimal"]


def test_order_up_to_policy_replay_and_capacity_reporting():
    data = inputs(
        ((2, 2, 2, 2, 2), (1, 1, 1, 1, 1)), initial=[2, 1], moq=2, capacity=10, decision_days=3
    )
    orders = order_up_to(data)
    assert orders.shape == (2, 3) and orders.sum() > 0
    trace = inventory_trace(data, orders)
    assert trace["capacity_violation_m3"] == 0
    score, rows = evaluate_policy(data, orders, policy="base", evaluation="forecast")
    assert len(rows) == 10
    assert score[1]["total_cost"] == pytest.approx(
        score[1]["holding_cost"] + score[1]["stockout_cost"] + score[1]["fixed_order_cost"]
    )
    export = purchase_orders(data, orders, policy="base", scenario_id="unit")
    assert (export.quantity >= 2).all()
    assert (
        (pd.to_datetime(export.arrival_date) - pd.to_datetime(export.order_date))
        .dt.days.eq(1)
        .all()
    )
    overflow = replace(data.with_demand(np.zeros_like(data.demand)), capacity={"S": 0.0})
    with pytest.raises(ValueError, match="capacity"):
        evaluate_policy(overflow, orders, policy="base", evaluation="realized")
    score, _ = evaluate_policy(
        overflow, orders, policy="base", evaluation="realized", allow_capacity_breach=True
    )
    assert not score[0]["capacity_feasible"] and score[0]["max_capacity_violation_m3"] > 0


def test_scenario_generation_is_preorigin_and_shock_preserves_state():
    origin = pd.Timestamp("2020-03-01")
    history = pd.DataFrame(
        {
            "unique_id": ["S/i0"] * 70,
            "ds": pd.date_range("2020-01-01", periods=70),
            "y": np.arange(70) % 4,
        }
    )
    data = inputs(moq=1, lead=2)
    econ = data.catalog.drop(columns=["unique_id", "store_id", "recent_std"])
    nodes = data.catalog[["unique_id", "store_id", "item_id"]]
    demand = np.ones((1, 28))
    before, meta, pipe = generate_scenario(history, econ, nodes, origin, demand)
    history.loc[history.ds > origin, "y"] = 10000
    after, changed, _ = generate_scenario(history, econ, nodes, origin, demand)
    np.testing.assert_array_equal(before.initial, after.initial)
    np.testing.assert_array_equal(before.receipts, after.receipts)
    pd.testing.assert_frame_equal(before.catalog, after.catalog)
    assert before.capacity == after.capacity and meta == changed
    assert pd.to_datetime(pipe.order_date).le(origin).all()
    shocked, _, delayed = generate_scenario(history, econ, nodes, origin, demand, lead_time_shock=4)
    np.testing.assert_array_equal(shocked.initial, before.initial)
    assert shocked.capacity == before.capacity
    np.testing.assert_array_equal(shocked.receipts[:, 4:], before.receipts[:, :-4])
    assert (
        (pd.to_datetime(delayed.arrival_date) - pd.to_datetime(pipe.arrival_date))
        .dt.days.eq(4)
        .all()
    )
    assert shocked.catalog.lead_time_days.iloc[0] == 6


def test_workflow_exports_validated_schedules_and_never_replans_on_actual(tmp_path):
    data = inputs(((0, 5, 5, 5, 5, 5, 5, 5, 5, 5),), decision_days=3, moq=1)
    config = OptimizationConfig(time_limit_seconds=5)
    policies = {MEDIAN: data.demand * 0.8}
    first, _ = run_phase(data, policies, config, tmp_path / "one", actual=data.demand)
    second, _ = run_phase(data, policies, config, tmp_path / "two", actual=data.demand * 2)
    a = np.load(tmp_path / "one/order_schedules.npz", allow_pickle=False)
    b = np.load(tmp_path / "two/order_schedules.npz", allow_pickle=False)
    for name in [MILP, MEDIAN]:
        np.testing.assert_array_equal(a[name], b[name])
    assert len(first) == 8 and set(second.evaluation) == {"forecast", "realized"}
    assert (tmp_path / "one/purchase_orders.csv").exists()
    bad = inputs(((0, 0),), initial=[10], capacity=1)
    with pytest.raises(PlanningFailure):
        run_phase(bad, {}, config, tmp_path / "bad")
    assert not (tmp_path / "bad/purchase_orders.csv").exists()


def test_invalid_inputs_forecast_keys_and_config_are_rejected():
    data = inputs()
    with pytest.raises(ValueError, match="nonnegative"):
        data.with_demand(np.array([[0.0, -1.0, 2.0]]))
    with pytest.raises(ValueError, match="capacity"):
        replace(data, capacity={})
    frame = pd.DataFrame({"unique_id": ["S/i0"], "ds": [data.dates[0]], "q90": [1.0]})
    with pytest.raises(ValueError, match="dates"):
        forecast_matrix(frame, ["S/i0"], data.dates, "q90")
    for kwargs in [
        {"service_quantile": 0.7},
        {"tail_days": 0},
        {"lead_time_shock_days": 5},
        {"capacity_multiplier": 0},
        {"time_limit_seconds": float("nan")},
    ]:
        with pytest.raises(ValueError):
            OptimizationConfig(**kwargs)


def test_item_relaxation_bound_and_coupled_capacity_fallback(tmp_path):
    from supply_chain_opt_engine.inventory_model import solve_store

    loose = inputs(((0, 2, 0), (0, 2, 0)), moq=3, fee=0, penalty=10, capacity=3)
    independent = solve_store(loose, time_limit=5, mip_gap=0, log_path=tmp_path / "loose.log")
    assert independent.diagnostics["strategy"] == "independent_items_jointly_capacity_feasible"
    assert independent.diagnostics["incumbent_objective"] == pytest.approx(0.4)
    assert independent.diagnostics["relative_gap"] == pytest.approx(0)
    tight = replace(loose, capacity={"S": 1.5})
    coupled = solve_store(tight, time_limit=5, mip_gap=0, log_path=tmp_path / "tight.log")
    assert coupled.diagnostics["strategy"] == "coupled_after_capacity_conflict"
    assert coupled.diagnostics["incumbent_objective"] == pytest.approx(20.2)
    assert coupled.inventory.sum(axis=0).max() <= 1.5 + 1e-8


def test_incremental_baseline_matches_daily_protection_and_capacity():
    data = inputs(((2, 2, 2, 2, 2),), initial=[1], decision_days=3, moq=1)
    np.testing.assert_array_equal(order_up_to(data), [[3, 1, 4]])
    tight = inputs(((0, 2, 0), (0, 2, 0)), moq=3, capacity=1.5)
    orders = order_up_to(tight)
    assert orders.sum() == 3
    assert inventory_trace(tight, orders)["capacity_violation_m3"] == 0


def test_lead_time_shock_prevents_earlier_fulfillment():
    base = inputs(((0, 5, 0),), moq=1, lead=1, fee=0, penalty=100)
    normal = solve_inventory(base, time_limit=5, mip_gap=0)
    shocked_catalog = base.catalog.copy()
    shocked_catalog["lead_time_days"] = 2
    delayed = solve_inventory(replace(base, catalog=shocked_catalog), time_limit=5, mip_gap=0)
    assert normal.unmet.sum() == pytest.approx(0)
    assert delayed.unmet.sum() == pytest.approx(5)
    assert delayed.orders.sum() == 0


def test_realized_capacity_flags_apply_to_each_reporting_window():
    data = inputs(((0, 0, 0, 0, 0),), decision_days=3, moq=1, capacity=0)
    scores, _ = evaluate_policy(
        data,
        np.array([[0, 0, 1]]),
        policy="test",
        evaluation="realized",
        allow_capacity_breach=True,
    )
    assert scores[0]["capacity_feasible"]
    assert not scores[1]["capacity_feasible"]
    assert scores[0]["max_capacity_violation_m3"] == 0
    assert scores[1]["max_capacity_violation_m3"] == 1
