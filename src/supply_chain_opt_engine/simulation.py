"""Physical lost-sales replay and deterministic forecast-only order-up-to policies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from supply_chain_opt_engine.inventory_data import InventoryInputs


def inventory_trace(data: InventoryInputs, orders: np.ndarray):
    orders = np.asarray(orders, dtype=float)
    if orders.shape != (data.n, data.decision_days) or not np.isfinite(orders).all():
        raise ValueError("Invalid order schedule shape or values.")
    if (orders < 0).any() or np.max(np.abs(orders - np.rint(orders))) > 1e-6:
        raise ValueError("Order schedule must contain nonnegative integer quantities.")
    moq = data.catalog.minimum_order_quantity.to_numpy()[:, None]
    if ((orders > 0) & (orders < moq)).any():
        raise ValueError("Order schedule violates MOQ.")
    arrival = data.receipts.astype(float).copy()
    for i, lead in enumerate(data.catalog.lead_time_days):
        for t in np.flatnonzero(orders[i]):
            day = t + int(lead)
            if day >= data.h:
                raise ValueError("An order arrives beyond the modeled horizon.")
            arrival[i, day] += orders[i, t]
    inventory = np.zeros_like(data.demand)
    unmet = np.zeros_like(data.demand)
    fulfilled = np.zeros_like(data.demand)
    stock = data.initial.astype(float).copy()
    for t in range(data.h):
        available = stock + arrival[:, t]
        fulfilled[:, t] = np.minimum(available, data.demand[:, t])
        unmet[:, t] = data.demand[:, t] - fulfilled[:, t]
        stock = available - fulfilled[:, t]
        inventory[:, t] = stock
    volume = data.catalog.unit_volume_m3.to_numpy()
    violation = 0.0
    for store, cap in data.capacity.items():
        mask = data.catalog.store_id.eq(store).to_numpy()
        used = volume[mask] @ inventory[mask]
        violation = max(violation, float(np.max(used - cap)))
    return {
        "arrivals": arrival,
        "inventory": inventory,
        "unmet": unmet,
        "fulfilled": fulfilled,
        "capacity_violation_m3": violation,
    }


def _candidate_stock(data, trace, i, arrival, quantity):
    stock = float(trace["inventory"][i, arrival - 1]) if arrival else data.initial[i]
    candidate = np.empty(data.h - arrival)
    for offset, day in enumerate(range(arrival, data.h)):
        stock = max(
            stock
            + trace["arrivals"][i, day]
            + (quantity if day == arrival else 0)
            - data.demand[i, day],
            0,
        )
        candidate[offset] = stock
    return candidate


def order_up_to(data: InventoryInputs, safety_stock=None):
    """Daily review, L+1 protection, MOQ, and greedy forecast-capacity admission.

    The entire schedule is frozen at the origin. Future realized demand never
    enters this planner. Store-item ID ordering resolves shared capacity conflicts. The final
    review covers all remaining tail demand because no later orders are allowed.
    """
    safety = np.zeros(data.n) if safety_stock is None else np.asarray(safety_stock, dtype=float)
    if safety.shape != (data.n,) or not np.isfinite(safety).all() or (safety < 0).any():
        raise ValueError("Safety stock must be finite nonnegative units for every series.")
    orders = np.zeros((data.n, data.decision_days), dtype=int)
    lead = data.catalog.lead_time_days.to_numpy(dtype=int)
    moq = data.catalog.minimum_order_quantity.to_numpy(dtype=int)
    volume = data.catalog.unit_volume_m3.to_numpy()
    trace = inventory_trace(data, orders)
    if trace["capacity_violation_m3"] > 1e-7:
        raise ValueError("Initial inventory/pipeline alone exceeds forecast end-of-day capacity.")
    stores = data.catalog.store_id.to_numpy()
    used_by_store = {}
    for store in data.capacity:
        mask = stores == store
        used_by_store[store] = volume[mask] @ trace["inventory"][mask]
    for t in range(data.decision_days):
        # Update only the affected leaf's future stock after accepting an order.
        previous = data.initial if t == 0 else trace["inventory"][:, t - 1]
        current_arrivals = trace["arrivals"][:, t]
        for i in range(data.n):
            arrival = t + lead[i]
            if arrival >= data.h:
                continue
            pending = trace["arrivals"][i, t + 1 : arrival + 1].sum()
            position = previous[i] + current_arrivals[i] + pending
            target_end = data.h if t == data.decision_days - 1 else arrival + 1
            target = data.demand[i, t:target_end].sum() + safety[i]
            desired = int(np.ceil(max(target - position, 0)))
            if desired == 0:
                continue
            desired = max(desired, int(moq[i]))
            store = stores[i]
            used = used_by_store[store]

            # Candidate stock changes only at this leaf. Binary search admits
            # the largest feasible integer quantity without arbitrary big-M.

            low, high = 0, desired
            while low < high:
                mid = (low + high + 1) // 2
                candidate = _candidate_stock(data, trace, i, arrival, mid)
                delta = (candidate - trace["inventory"][i, arrival:]) * volume[i]
                if np.max(used[arrival:] + delta - data.capacity[store]) <= 1e-8:
                    low = mid
                else:
                    high = mid - 1
            if low >= moq[i]:
                orders[i, t] = low
                candidate = _candidate_stock(data, trace, i, arrival, low)
                used[arrival:] += (candidate - trace["inventory"][i, arrival:]) * volume[i]
                trace["inventory"][i, arrival:] = candidate
                trace["arrivals"][i, arrival] += low
    return orders


def evaluate_policy(data, orders, *, policy, evaluation, allow_capacity_breach=False):
    trace = inventory_trace(data, orders)
    if trace["capacity_violation_m3"] > 1e-5 and not allow_capacity_breach:
        raise ValueError(f"{policy} exceeds capacity during {evaluation} replay.")
    c = data.catalog
    fee = (orders > 0) * c.fixed_order_cost.to_numpy()[:, None]
    holding = trace["inventory"] * c.daily_holding_cost.to_numpy()[:, None]
    stockout = trace["unmet"] * c.stockout_penalty.to_numpy()[:, None]
    capital = trace["inventory"] * c.unit_purchase_cost.to_numpy()[:, None]
    summaries = []
    for scope, end in [("decision_window", data.decision_days), ("including_tail", data.h)]:
        demand = float(data.demand[:, :end].sum())
        fulfilled = float(trace["fulfilled"][:, :end].sum())
        h = float(holding[:, :end].sum())
        u = float(stockout[:, :end].sum())
        f = float(fee[:, : min(end, data.decision_days)].sum())
        scope_violation = 0.0
        for store, capacity in data.capacity.items():
            mask = c.store_id.eq(store).to_numpy()
            used = c.unit_volume_m3.to_numpy()[mask] @ trace["inventory"][mask, :end]
            scope_violation = max(scope_violation, float(np.max(used - capacity)))
        summaries.append(
            {
                "policy": policy,
                "evaluation": evaluation,
                "scope": scope,
                "days": end,
                "demand_units": demand,
                "fulfilled_units": fulfilled,
                "fill_rate": fulfilled / demand if demand else None,
                "holding_cost": h,
                "stockout_cost": u,
                "fixed_order_cost": f,
                "total_cost": h + u + f,
                "average_inventory_value": float(capital[:, :end].sum() / end),
                "order_units": int(orders.sum()),
                "order_count": int((orders > 0).sum()),
                "terminal_inventory_units": float(trace["inventory"][:, end - 1].sum()),
                "max_capacity_violation_m3": scope_violation,
                "capacity_feasible": scope_violation <= 1e-5,
            }
        )
    scheduled = np.zeros((data.n, data.h), dtype=int)
    scheduled[:, : data.decision_days] = orders
    fixed = np.zeros((data.n, data.h))
    fixed[:, : data.decision_days] = fee
    rows = pd.DataFrame(
        {
            "unique_id": np.repeat(c.unique_id.to_numpy(), data.h),
            "ds": np.tile(data.dates, data.n),
            "demand": data.demand.reshape(-1),
            "arrivals": trace["arrivals"].reshape(-1),
            "order_quantity": scheduled.reshape(-1),
            "fixed_order_cost": fixed.reshape(-1),
            "fulfilled": trace["fulfilled"].reshape(-1),
            "unmet": trace["unmet"].reshape(-1),
            "inventory": trace["inventory"].reshape(-1),
            "holding_cost": holding.reshape(-1),
            "stockout_cost": stockout.reshape(-1),
            "inventory_value": capital.reshape(-1),
            "policy": policy,
            "evaluation": evaluation,
        }
    )
    return summaries, rows


def purchase_orders(data, orders, *, policy, scenario_id):
    # Export only after model/schedule validation. No zero orders or tail decisions.
    inventory_trace(data, orders)
    rows = []
    for i, t in zip(*np.nonzero(orders), strict=True):
        item = data.catalog.iloc[i]
        rows.append(
            {
                "scenario_id": scenario_id,
                "policy": policy,
                "unique_id": item.unique_id,
                "store_id": item.store_id,
                "item_id": item.item_id,
                "order_date": str(data.dates[t].date()),
                "arrival_date": str(data.dates[t + int(item.lead_time_days)].date()),
                "quantity": int(orders[i, t]),
                "unit_purchase_cost": item.unit_purchase_cost,
                "purchase_value": float(orders[i, t] * item.unit_purchase_cost),
                "fixed_order_cost": item.fixed_order_cost,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "scenario_id",
            "policy",
            "unique_id",
            "store_id",
            "item_id",
            "order_date",
            "arrival_date",
            "quantity",
            "unit_purchase_cost",
            "purchase_value",
            "fixed_order_cost",
        ],
    )
