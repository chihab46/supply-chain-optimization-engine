"""Pinned, checksummed data views for the local planning dashboard."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from supply_chain_opt_engine.hierarchy import Hierarchy
from supply_chain_opt_engine.history import load_history
from supply_chain_opt_engine.m5 import checksum
from supply_chain_opt_engine.optimization import MILP, checked_bundle

ALL = "all"
POLICY_LABELS = {
    MILP: "MinT quantile MILP",
    "lightgbm_p50_order_up_to": "Global P50 order-up-to",
    "lightgbm_p90_order_up_to": "Global P90 order-up-to",
    "seasonal_naive_p50_order_up_to": "SeasonalNaive order-up-to",
    "autoets_static_safety_stock": "AutoETS static safety stock",
}


def verify_artifact(directory, document, name):
    if (
        name not in document["artifact_sha256"]
        or checksum(directory / name) != document["artifact_sha256"][name]
    ):
        raise ValueError(f"Saved artifact failed verification: {name}")
    return directory / name


class DashboardRepository:
    """One immutable forecast snapshot for the lifetime of an app instance."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.shap_importance = lru_cache(maxsize=2)(self.shap_importance)
        self.result = lru_cache(maxsize=4)(self.result)
        self.recpath, self.rec = checked_bundle(
            self.root, "artifacts/forecasts/reconciliation", "reconciled"
        )
        self.globalpath = (
            self.root / "artifacts/models/lightgbm" / self.rec["upstream_run_id"] / "manifest.json"
        )
        if checksum(self.globalpath) != self.rec["upstream_manifest_sha256"]:
            raise ValueError("Forecast model lineage differs from reconciliation.")
        self.global_manifest = json.loads(self.globalpath.read_text())
        self.basepath, self.baseline = checked_bundle(
            self.root, "artifacts/metrics/baselines", "backtested"
        )
        self.nodes = pd.read_csv(self.recpath.parent / "hierarchy_nodes.csv")
        matrix = np.load(self.recpath.parent / "hierarchy.npz", allow_pickle=False)
        self.hierarchy = Hierarchy(self.nodes, matrix["S"])
        self.leaves = self.nodes[self.nodes.level.eq("store_item")].copy()
        ingest = (
            self.root / "data/interim" / self.rec["source"]["ingestion_run_id"] / "manifest.json"
        )
        if checksum(ingest) != self.rec["source"]["manifest_sha256"]:
            raise ValueError("History lineage differs from reconciliation.")
        history, self.source = load_history(self.root, manifest_path=ingest)
        self.history, self.history_dates = self.hierarchy.history_matrix(history)
        self.node_indices = dict(zip(self.nodes.unique_id, self.nodes.index, strict=True))
        historical = pd.read_parquet(self.recpath.parent / "predictions.parquet")
        primary = self.rec["config"]["primary_method"]
        self.forecasts = {
            "latest": pd.read_parquet(self.recpath.parent / "latest_forecasts.parquet").query(
                "model == @primary"
            )
        }
        self.origins = {"latest": pd.Timestamp(self.rec["latest_origin"])}
        self.origin_labels = {"latest": f"Latest · {self.origins['latest']:%b %d, %Y}"}
        for fold in reversed(self.rec["folds"]):
            key = f"fold_{fold['fold_id']}"
            self.forecasts[key] = historical[
                historical.fold_id.eq(fold["fold_id"]) & historical.model.eq(primary)
            ]
            self.origins[key] = pd.Timestamp(fold["origin"])
            role = (
                "Holdout" if fold["role"] == "holdout" else fold["season"].replace("_", " ").title()
            )
            self.origin_labels[key] = f"{role} · {self.origins[key]:%b %d, %Y}"
        self.holdout = next(f for f in self.rec["folds"] if f["role"] == "holdout")
        self.phase_labels = {
            "holdout": f"Holdout · {pd.Timestamp(self.holdout['origin']):%b %d, %Y}",
            "latest": self.origin_labels["latest"],
        }
        self.reference_runs = []
        for p in (self.root / "artifacts/orders/optimization").glob("*/manifest.json"):
            d = json.loads(p.read_text())
            if (
                d.get("status") == "optimized"
                and d.get("reconciliation_run_id") == self.rec["run_id"]
            ):
                self.reference_runs.append((p, d))
        self.reference_runs.sort(key=lambda pair: pair[0].stat().st_mtime, reverse=True)

    def category_options(self, store):
        if store == ALL:
            return []
        return sorted(self.leaves.loc[self.leaves.store_id.eq(store), "cat_id"].unique())

    def item_options(self, store, category):
        part = self.leaves[self.leaves.store_id.eq(store) & self.leaves.cat_id.eq(category)]
        return [{"label": r.item_id, "value": r.unique_id} for r in part.itertuples()]

    def node_id(self, store=ALL, category=ALL, item=ALL):
        if store == ALL:
            if category != ALL or item != ALL:
                raise ValueError("Select a store before a category or item.")
            return "Total"
        if category == ALL:
            if item != ALL:
                raise ValueError("Select a category before an item.")
            node = f"Store/{store}"
        elif item == ALL:
            node = f"StoreCategory/{store}/{category}"
        else:
            part = self.leaves[self.leaves.unique_id.eq(item)]
            if len(part) != 1 or part.store_id.iloc[0] != store or part.cat_id.iloc[0] != category:
                raise ValueError("Selected item does not belong to this store and category.")
            node = item
        if node not in self.node_indices:
            raise ValueError("Unknown hierarchy selection.")
        return node

    def forecast_view(self, origin_key, store=ALL, category=ALL, item=ALL):
        if origin_key not in self.origins:
            raise ValueError("Unknown forecast origin.")
        node = self.node_id(store, category, item)
        origin = self.origins[origin_key]
        forecast = (
            self.forecasts[origin_key].loc[lambda f: f.unique_id.eq(node)].sort_values("ds").copy()
        )
        mask = (self.history_dates >= origin - pd.Timedelta(days=41)) & (
            self.history_dates <= origin + pd.Timedelta(days=28)
        )
        observed = pd.DataFrame(
            {"ds": self.history_dates[mask], "y": self.history[self.node_indices[node], mask]}
        )
        return node, origin, observed, forecast

    def shap_importance(self, origin_key):
        # SHAP is model-wide. Tuning folds have no frozen SHAP sample and must
        # not inherit a later model's explanation silently.
        role = (
            "latest"
            if origin_key == "latest"
            else "holdout"
            if origin_key == f"fold_{self.holdout['fold_id']}"
            else None
        )
        if role is None:
            return None, None
        filename = f"{role}_shap_importance.csv"
        path = verify_artifact(self.globalpath.parent, self.global_manifest, filename)
        frame = pd.read_csv(path).sort_values("mean_absolute_shap", ascending=False).head(7)
        return frame, role

    def reference(self, phase, config):
        for path, document in self.reference_runs:
            if phase not in [p["phase"] for p in document["phases"]]:
                continue
            if all(document["config"].get(k) == v for k, v in config.items()):
                return path
        return None

    def result(self, manifest_path, phase):
        path = Path(manifest_path).resolve()
        allowed = (self.root / "artifacts/orders").resolve()
        if not path.is_relative_to(allowed):
            raise ValueError("Scenario result must be a local inventory bundle.")
        document = json.loads(path.read_text())
        if (
            document.get("status") != "optimized"
            or document["reconciliation_run_id"] != self.rec["run_id"]
        ):
            raise ValueError("Scenario results do not match the active forecast snapshot.")
        if checksum(self.recpath) != document["reconciliation_manifest_sha256"]:
            raise ValueError("Scenario forecast lineage failed verification.")
        for name in [
            "policy_metrics.csv",
            f"{phase}/inventory_trace.parquet",
            f"{phase}/purchase_orders.csv",
            f"{phase}/solver_diagnostics.json",
        ]:
            verify_artifact(path.parent, document, name)
        metrics = pd.read_csv(path.parent / "policy_metrics.csv")
        metrics = metrics[metrics.phase.eq(phase)].copy()
        if metrics.empty:
            raise ValueError("The selected phase is absent from this result.")
        trace = pd.read_parquet(path.parent / phase / "inventory_trace.parquet")
        orders = pd.read_csv(path.parent / phase / "purchase_orders.csv")
        orders = orders[orders.policy.eq(MILP)].copy()
        solver = json.loads((path.parent / phase / "solver_diagnostics.json").read_text())
        if not all(d.get("has_feasible_solution") for d in solver):
            raise ValueError("No validated feasible solver result is available.")
        total = sum(d["incumbent_objective"] for d in solver)
        bounds = [d.get("objective_bound") for d in solver]
        lower = sum(bounds) if all(b is not None and np.isfinite(b) for b in bounds) else None
        return {
            "manifest": document,
            "metrics": metrics,
            "trace": trace,
            "orders": orders,
            "gap": (
                max(0.0, total - lower) / max(abs(total), 1e-10) if lower is not None else None
            ),
            "path": path,
        }
