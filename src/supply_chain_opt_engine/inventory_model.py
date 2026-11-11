"""Pyomo inventory MILP, bounded order quantities and guarded HiGHS extraction."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers import Highs

from supply_chain_opt_engine.inventory_data import InventoryInputs


@dataclass
class InventorySolution:
    orders: np.ndarray | None
    inventory: np.ndarray | None
    unmet: np.ndarray | None
    diagnostics: dict


def order_bounds(data: InventoryInputs):
    """Finite bounds from remaining demand, MOQ and storage on the arrival day."""
    c = data.catalog
    lead = c.lead_time_days.to_numpy(dtype=int)
    moq = c.minimum_order_quantity.to_numpy(dtype=int)
    volume = c.unit_volume_m3.to_numpy()
    capacity = c.store_id.map(data.capacity).to_numpy()
    penalty = c.stockout_penalty.to_numpy()
    fee = c.fixed_order_cost.to_numpy()
    bounds = np.zeros((data.n, data.decision_days), dtype=int)
    for i in range(data.n):
        for t in range(data.decision_days):
            arrival = t + lead[i]
            if arrival >= data.h:
                continue
            remaining = float(data.demand[i, arrival:].sum())
            # Without terminal minimum stock, an order cannot save more than all
            # remaining lost-sales penalties. Holding and capacity only reduce its value.
            if remaining * penalty[i] <= fee[i] or remaining == 0:
                continue
            useful = max(moq[i], int(np.ceil(remaining)))
            storage = int(np.floor(capacity[i] / volume[i] + data.demand[i, arrival] + 1e-9))
            if storage >= moq[i]:
                bounds[i, t] = min(useful, storage)
    return bounds


def build_inventory_model(data: InventoryInputs):
    c = data.catalog
    lead = c.lead_time_days.to_numpy(dtype=int)
    moq = c.minimum_order_quantity.to_numpy(dtype=int)
    volume = c.unit_volume_m3.to_numpy()
    holding = c.daily_holding_cost.to_numpy()
    penalty = c.stockout_penalty.to_numpy()
    fee = c.fixed_order_cost.to_numpy()
    upper = order_bounds(data)
    m = pyo.ConcreteModel("lost_sales_replenishment")
    m.skus = pyo.RangeSet(0, data.n - 1)
    m.days = pyo.RangeSet(0, data.h - 1)
    m.order_index = pyo.Set(dimen=2, initialize=list(zip(*np.nonzero(upper), strict=True)))
    m.shortage_index = pyo.Set(
        dimen=2, initialize=list(zip(*np.nonzero(data.demand > 0), strict=True))
    )
    # Bound stock using both per-item capacity and reachable cumulative supply.
    imax = np.empty((data.n, data.h))
    for i in range(data.n):
        incoming = np.zeros(data.h)
        for t in range(data.decision_days):
            if upper[i, t]:
                incoming[t + lead[i]] += upper[i, t]
        reachable = data.initial[i] + np.cumsum(data.receipts[i] + incoming)
        imax[i] = np.minimum(data.capacity[c.store_id.iloc[i]] / volume[i], reachable)
    m.order = pyo.Var(
        m.order_index, domain=pyo.NonNegativeIntegers, bounds=lambda _, i, t: (0, int(upper[i, t]))
    )
    m.trigger = pyo.Var(m.order_index, domain=pyo.Binary)
    m.inventory = pyo.Var(
        m.skus, m.days, domain=pyo.NonNegativeReals, bounds=lambda _, i, t: (0, float(imax[i, t]))
    )
    m.unmet = pyo.Var(
        m.skus,
        m.days,
        domain=pyo.NonNegativeReals,
        bounds=lambda _, i, t: (0, float(data.demand[i, t])),
    )
    m.shortage = pyo.Var(m.shortage_index, domain=pyo.Binary)

    def balance(_, i, t):
        previous = data.initial[i] if t == 0 else m.inventory[i, t - 1]
        arrival = m.order[i, t - lead[i]] if (i, t - lead[i]) in m.order_index else 0
        return m.inventory[i, t] - m.unmet[i, t] == (
            previous + float(data.receipts[i, t]) + arrival - float(data.demand[i, t])
        )

    m.balance = pyo.Constraint(m.skus, m.days, rule=balance)
    m.moq = pyo.Constraint(
        m.order_index, rule=lambda _, i, t: m.order[i, t] >= int(moq[i]) * m.trigger[i, t]
    )
    m.order_limit = pyo.Constraint(
        m.order_index, rule=lambda _, i, t: m.order[i, t] <= int(upper[i, t]) * m.trigger[i, t]
    )
    m.no_rationing = pyo.Constraint(
        m.shortage_index,
        rule=lambda _, i, t: m.inventory[i, t] <= float(imax[i, t]) * (1 - m.shortage[i, t]),
    )
    m.lost_sales = pyo.Constraint(
        m.shortage_index,
        rule=lambda _, i, t: m.unmet[i, t] <= float(data.demand[i, t]) * m.shortage[i, t],
    )
    m.store_days = pyo.Set(
        dimen=2, initialize=[(s, t) for s in sorted(data.capacity) for t in range(data.h)]
    )
    groups = {s: np.flatnonzero(c.store_id.eq(s)) for s in data.capacity}
    m.capacity = pyo.Constraint(
        m.store_days,
        rule=lambda _, s, t: (
            sum(float(volume[i]) * m.inventory[i, t] for i in groups[s]) <= data.capacity[s]
        ),
    )
    m.cost = pyo.Objective(
        expr=(
            sum(float(fee[i]) * m.trigger[i, t] for i, t in m.order_index)
            + sum(
                float(holding[i]) * m.inventory[i, t] + float(penalty[i]) * m.unmet[i, t]
                for i in range(data.n)
                for t in range(data.h)
            )
        ),
        sense=pyo.minimize,
    )
    return m


def solution_acceptable(termination, objective):
    return (
        termination
        in {
            TerminationCondition.optimal,
            TerminationCondition.maxTimeLimit,
            TerminationCondition.maxIterations,
            TerminationCondition.objectiveLimit,
        }
        and objective is not None
        and np.isfinite(objective)
    )


def validate_solution(data, orders, inventory, unmet, objective=None, atol=1e-5):
    """Independently enforce physical fulfillment, integrality and objective accounting."""
    from supply_chain_opt_engine.simulation import inventory_trace

    if orders.shape != (data.n, data.decision_days) or not np.isfinite(orders).all():
        raise ValueError("Invalid extracted order schedule.")
    if (orders < -atol).any() or np.max(np.abs(orders - np.rint(orders))) > atol:
        raise ValueError("Order quantities must be nonnegative integers.")
    orders = np.rint(orders).astype(int)
    if (orders > order_bounds(data)).any():
        raise ValueError("Orders exceed their finite planning bounds.")
    moq = data.catalog.minimum_order_quantity.to_numpy()[:, None]
    if ((orders > 0) & (orders < moq)).any():
        raise ValueError("A positive order violates MOQ.")
    trace = inventory_trace(data, orders)
    if not np.allclose(trace["inventory"], inventory, rtol=1e-8, atol=atol):
        raise ValueError("Extracted inventory violates work-conserving flow balance.")
    if not np.allclose(trace["unmet"], unmet, rtol=1e-8, atol=atol):
        raise ValueError("Extracted lost sales violate physical fulfillment.")
    if trace["capacity_violation_m3"] > atol:
        raise ValueError("Extracted inventory violates store capacity.")
    value = float(
        (trace["inventory"] * data.catalog.daily_holding_cost.to_numpy()[:, None]).sum()
        + (trace["unmet"] * data.catalog.stockout_penalty.to_numpy()[:, None]).sum()
        + ((orders > 0) * data.catalog.fixed_order_cost.to_numpy()[:, None]).sum()
    )
    if objective is not None and not np.isclose(value, objective, rtol=1e-7, atol=atol):
        raise ValueError("Solver objective differs from independently recomputed costs.")
    return {
        "objective_recomputed": value,
        "max_capacity_violation_m3": trace["capacity_violation_m3"],
        "max_inventory_error": float(np.max(np.abs(trace["inventory"] - inventory))),
        "max_unmet_error": float(np.max(np.abs(trace["unmet"] - unmet))),
    }


def solve_inventory(
    data,
    *,
    time_limit=60.0,
    mip_gap=0.01,
    seed=42,
    threads=1,
    log_path=None,
    solver=None,
    start_orders=None,
):
    start = time.perf_counter()
    m = build_inventory_model(data)
    build_seconds = time.perf_counter() - start
    solver = Highs() if solver is None else solver
    solver.config.load_solution = False
    solver.config.log_level = logging.DEBUG
    solver.config.time_limit = time_limit
    solver.config.mip_gap = mip_gap
    solver.highs_options.update(threads=threads, random_seed=seed, mip_feasibility_tolerance=1e-7)
    if log_path is not None:
        solver.highs_options["log_file"] = str(log_path)
    from supply_chain_opt_engine.simulation import inventory_trace

    seed_orders = (
        np.zeros((data.n, data.decision_days), dtype=int) if start_orders is None else start_orders
    )
    seed_trace = inventory_trace(data, seed_orders)
    if seed_trace["capacity_violation_m3"] <= 1e-7:
        for i, t in m.order_index:
            m.order[i, t].value = int(seed_orders[i, t])
            m.trigger[i, t].value = int(seed_orders[i, t] > 0)
        for i in range(data.n):
            for t in range(data.h):
                m.inventory[i, t].value = float(seed_trace["inventory"][i, t])
                m.unmet[i, t].value = float(seed_trace["unmet"][i, t])
        for i, t in m.shortage_index:
            m.shortage[i, t].value = int(seed_trace["unmet"][i, t] > 0)
        solver.config.warmstart = True
    solve_start = time.perf_counter()
    result = solver.solve(m)
    incumbent = result.best_feasible_objective
    bound = result.best_objective_bound
    if bound is not None and not np.isfinite(bound):
        bound = None
    if incumbent is not None and not np.isfinite(incumbent):
        incumbent = None
    status = result.termination_condition
    gap = None
    if incumbent is not None and bound is not None and np.isfinite([incumbent, bound]).all():
        gap = max(0.0, float(incumbent - bound)) / max(abs(incumbent), 1e-10)
    diag = {
        "termination": status.name,
        "incumbent_objective": incumbent,
        "objective_bound": bound,
        "relative_gap": gap,
        "build_seconds": build_seconds,
        "solve_seconds": time.perf_counter() - solve_start,
        "variables": sum(1 for _ in m.component_data_objects(pyo.Var)),
        "constraints": sum(1 for _ in m.component_data_objects(pyo.Constraint)),
        "order_variables": len(m.order_index),
        "has_feasible_solution": False,
    }
    if not solution_acceptable(status, incumbent):
        return InventorySolution(None, None, None, diag)
    result.solution_loader.load_vars()
    orders = np.zeros((data.n, data.decision_days))
    for i, t in m.order_index:
        orders[i, t] = pyo.value(m.order[i, t])
    inventory = np.array(
        [[pyo.value(m.inventory[i, t]) for t in range(data.h)] for i in range(data.n)]
    )
    unmet = np.array([[pyo.value(m.unmet[i, t]) for t in range(data.h)] for i in range(data.n)])
    diag.update(validate_solution(data, orders, inventory, unmet, incumbent))
    diag["has_feasible_solution"] = True
    diag["proven_optimal"] = status == TerminationCondition.optimal and (gap or 0) <= 1e-8
    # "optimal" is HiGHS termination within configured MIP tolerances. Preserve the bound/gap.
    return InventorySolution(np.rint(orders).astype(int), inventory, unmet, diag)


def solve_store(data, *, time_limit=60.0, mip_gap=0.01, seed=42, threads=1, log_path=None):
    """Use separable item bounds only when the combined schedule respects capacity.

    Removing cross-item capacity is a relaxation. Its summed objective bounds
    remain valid for the coupled store problem. A joint feasible incumbent plus
    this lower bound certifies the reported gap. Capacity conflicts trigger the
    original coupled MILP instead.
    """
    from dataclasses import replace

    from supply_chain_opt_engine.simulation import inventory_trace

    if data.n == 1:
        return solve_inventory(
            data,
            time_limit=time_limit,
            mip_gap=mip_gap,
            seed=seed,
            threads=threads,
            log_path=log_path,
        )
    zero = np.zeros((data.n, data.decision_days), dtype=int)
    trace = inventory_trace(data, zero)
    if trace["capacity_violation_m3"] > 1e-7:
        return solve_inventory(
            data,
            time_limit=time_limit,
            mip_gap=mip_gap,
            seed=seed,
            threads=threads,
            log_path=log_path,
        )
    forced_cost = (trace["inventory"] * data.catalog.daily_holding_cost.to_numpy()[:, None]).sum(
        axis=1
    ) + (trace["unmet"] * data.catalog.stockout_penalty.to_numpy()[:, None]).sum(axis=1)
    objectives = forced_cost.copy()
    bounds = forced_cost.copy()
    orders = zero.copy()
    active = np.flatnonzero(order_bounds(data).any(axis=1))
    children = []
    remaining = time_limit
    for counter, i in enumerate(active):
        item = replace(
            data,
            catalog=data.catalog.iloc[[i]].reset_index(drop=True),
            demand=data.demand[[i]],
            initial=data.initial[[i]],
            receipts=data.receipts[[i]],
        )
        allowance = max(0.001, remaining / (len(active) - counter))
        result = solve_inventory(
            item,
            time_limit=allowance,
            mip_gap=mip_gap,
            seed=seed,
            threads=threads,
            log_path=log_path,
        )
        children.append({"unique_id": data.catalog.unique_id.iloc[i], **result.diagnostics})
        remaining -= result.diagnostics["solve_seconds"]
        if result.orders is None:
            fallback = solve_inventory(
                data,
                time_limit=max(0.001, remaining),
                mip_gap=mip_gap,
                seed=seed,
                threads=threads,
                log_path=log_path,
            )
            fallback.diagnostics.update(
                strategy="coupled_after_missing_item_incumbent", item_solves=children
            )
            fallback.diagnostics["solve_seconds"] += sum(c["solve_seconds"] for c in children)
            return fallback
        orders[i] = result.orders[0]
        objectives[i] = result.diagnostics["incumbent_objective"]
        lower = result.diagnostics["objective_bound"]
        bounds[i] = max(0.0, lower) if lower is not None and np.isfinite(lower) else 0.0
    # Spend unused budget on the largest remaining item bounds, rather than
    # losing it when easy items finish before their original allowance.
    pending = sorted(active, key=lambda i: objectives[i] - bounds[i], reverse=True)
    for counter, i in enumerate(pending):
        global_gap = max(0.0, float(objectives.sum() - bounds.sum())) / max(
            float(objectives.sum()), 1e-10
        )
        if global_gap <= mip_gap or remaining <= 0.01:
            break
        if objectives[i] - bounds[i] <= 1e-8:
            continue
        item = replace(
            data,
            catalog=data.catalog.iloc[[i]].reset_index(drop=True),
            demand=data.demand[[i]],
            initial=data.initial[[i]],
            receipts=data.receipts[[i]],
        )
        gaps = sum(max(0.0, objectives[j] - bounds[j]) for j in pending[counter:])
        allowance = max(0.001, remaining * (objectives[i] - bounds[i]) / max(gaps, 1e-10))
        refined = solve_inventory(
            item,
            time_limit=allowance,
            mip_gap=mip_gap,
            seed=seed,
            threads=threads,
            log_path=log_path,
            start_orders=orders[[i]],
        )
        children.append(
            {"unique_id": data.catalog.unique_id.iloc[i], "refinement": True, **refined.diagnostics}
        )
        remaining -= refined.diagnostics["solve_seconds"]
        lower = refined.diagnostics["objective_bound"]
        if lower is not None:
            bounds[i] = max(bounds[i], lower)
        if (
            refined.orders is not None
            and refined.diagnostics["incumbent_objective"] < objectives[i]
        ):
            orders[i] = refined.orders[0]
            objectives[i] = refined.diagnostics["incumbent_objective"]
    combined = inventory_trace(data, orders)
    if combined["capacity_violation_m3"] > 1e-7:
        fallback = solve_inventory(
            data,
            time_limit=max(0.001, remaining),
            mip_gap=mip_gap,
            seed=seed,
            threads=threads,
            log_path=log_path,
        )
        fallback.diagnostics.update(
            strategy="coupled_after_capacity_conflict", item_solves=children
        )
        fallback.diagnostics["solve_seconds"] += sum(c["solve_seconds"] for c in children)
        return fallback
    obj = float(objectives.sum())
    bound = float(bounds.sum())
    gap = max(0.0, obj - bound) / max(abs(obj), 1e-10)
    checks = validate_solution(data, orders, combined["inventory"], combined["unmet"], obj)
    diagnostics = {
        "strategy": "independent_items_jointly_capacity_feasible",
        "termination": "within_gap_tolerance" if gap <= mip_gap else "feasible_limit",
        "incumbent_objective": obj,
        "objective_bound": bound,
        "relative_gap": gap,
        "has_feasible_solution": True,
        "proven_optimal": gap <= 1e-8,
        "solve_seconds": sum(c["solve_seconds"] for c in children),
        "build_seconds": sum(c["build_seconds"] for c in children),
        "active_items": len(active),
        "fixed_zero_order_items": data.n - len(active),
        "item_solves": children,
        **checks,
    }
    return InventorySolution(orders, combined["inventory"], combined["unmet"], diagnostics)
