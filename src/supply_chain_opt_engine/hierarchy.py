"""Stable retail hierarchy, summing matrix and numerical coherence checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEVELS = ("total", "store", "store_category", "store_item")


@dataclass
class Hierarchy:
    nodes: pd.DataFrame
    S: np.ndarray

    @classmethod
    def from_catalog(cls, catalog: pd.DataFrame):
        columns = ["unique_id", "store_id", "cat_id", "item_id"]
        if set(columns) - set(catalog) or catalog[columns].isna().any().any():
            raise ValueError(
                "Hierarchy catalog requires complete series, store, category and item IDs."
            )
        leaves = catalog[columns].drop_duplicates().sort_values("unique_id").reset_index(drop=True)
        if (
            leaves.empty
            or leaves.unique_id.duplicated().any()
            or leaves.duplicated(["store_id", "item_id"]).any()
        ):
            raise ValueError("Each hierarchy leaf must identify exactly one store-item pair.")
        if leaves.groupby("item_id").cat_id.nunique().gt(1).any():
            raise ValueError("Items cannot belong to different categories across stores.")
        rows = [{"unique_id": "Total", "level": "total", "parent_id": None}]
        for store in sorted(leaves.store_id.unique()):
            rows.append(
                {
                    "unique_id": f"Store/{store}",
                    "level": "store",
                    "parent_id": "Total",
                    "store_id": store,
                }
            )
        for store, category in sorted(
            set(leaves[["store_id", "cat_id"]].itertuples(index=False, name=None))
        ):
            rows.append(
                {
                    "unique_id": f"StoreCategory/{store}/{category}",
                    "level": "store_category",
                    "parent_id": f"Store/{store}",
                    "store_id": store,
                    "cat_id": category,
                }
            )
        for row in leaves.itertuples(index=False):
            rows.append(
                {
                    "unique_id": row.unique_id,
                    "level": "store_item",
                    "parent_id": f"StoreCategory/{row.store_id}/{row.cat_id}",
                    "store_id": row.store_id,
                    "cat_id": row.cat_id,
                    "item_id": row.item_id,
                }
            )
        nodes = pd.DataFrame(rows)
        if nodes.unique_id.duplicated().any():
            raise ValueError("Leaf IDs collide with reserved aggregate-node IDs.")
        S = np.zeros((len(nodes), len(leaves)), dtype=float)
        for index, row in nodes.iterrows():
            if row.level == "total":
                mask = np.ones(len(leaves), dtype=bool)
            elif row.level == "store":
                mask = leaves.store_id.eq(row.store_id).to_numpy()
            elif row.level == "store_category":
                mask = (leaves.store_id.eq(row.store_id) & leaves.cat_id.eq(row.cat_id)).to_numpy()
            else:
                mask = leaves.unique_id.eq(row.unique_id).to_numpy()
            S[index, mask] = 1
        return cls(nodes, S)

    @property
    def n_bottom(self):
        return self.S.shape[1]

    @property
    def bottom_ids(self):
        return self.nodes.unique_id.iloc[-self.n_bottom :].tolist()

    @property
    def tags(self):
        return {level: np.flatnonzero(self.nodes.level.eq(level)) for level in LEVELS}

    def aggregate(self, bottom: np.ndarray) -> np.ndarray:
        bottom = np.asarray(bottom, dtype=float)
        if bottom.shape[0] != self.n_bottom or not np.isfinite(bottom).all():
            raise ValueError("Bottom array must be finite and ordered by the hierarchy leaf index.")
        matrix = bottom.reshape(self.n_bottom, -1)
        # The bottom block is identity; avoid multiplying its structural zeros
        # when reconciling thousands of joint paths.
        aggregate = np.vstack([self.S[: -self.n_bottom] @ matrix, matrix])
        return aggregate.reshape((len(self.nodes), *bottom.shape[1:]))

    def coherence_error(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        if values.shape[0] != len(self.nodes) or not np.isfinite(values).all():
            raise ValueError("Hierarchy values have invalid node count or nonfinite entries.")
        return float(np.max(np.abs(values - self.aggregate(values[-self.n_bottom :]))))

    def assert_coherent(self, values: np.ndarray, atol: float = 1e-8):
        if not np.allclose(values, self.aggregate(values[-self.n_bottom :]), rtol=1e-10, atol=atol):
            raise ValueError("Forecasts do not sum coherently from leaves to aggregate nodes.")

    def history_matrix(self, history: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
        pivot = history.pivot(index="unique_id", columns="ds", values="y").reindex(self.bottom_ids)
        if pivot.isna().any().any():
            raise ValueError("The hierarchy needs a complete aligned daily history at every leaf.")
        return self.aggregate(pivot.to_numpy()), pd.DatetimeIndex(pivot.columns)
