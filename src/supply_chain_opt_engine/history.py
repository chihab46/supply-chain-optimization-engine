"""Read the scoped sales panel with series-selection provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text

from supply_chain_opt_engine.evaluation import validate_history
from supply_chain_opt_engine.m5 import checksum


def ingestion_manifest(root: Path, path: Path | None = None) -> tuple[Path, dict]:
    if path is None:
        candidates = []
        for candidate in (root / "data/interim").glob("*/manifest.json"):
            document = json.loads(candidate.read_text())
            if document.get("status") == "loaded":
                candidates.append(candidate)
        if not candidates:
            raise ValueError(
                "No committed ingestion manifest found. Load the scoped dataset first."
            )
        path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    document = json.loads(path.read_text())
    if document.get("status") != "loaded":
        raise ValueError("Backtesting requires a committed ingestion manifest with status loaded.")
    for name in ["selected_series.csv", "fact_daily_sales.csv"]:
        if checksum(path.parent / name) != document["artifact_sha256"][name]:
            raise ValueError(f"Ingestion artifact checksum mismatch: {name}")
    return path, document


def load_history(
    root: Path,
    engine: Engine | None = None,
    manifest_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Use PostgreSQL by default, or the exact checksummed CSVs committed by ingestion."""
    path, manifest = ingestion_manifest(root, manifest_path)
    selected = pd.read_csv(path.parent / "selected_series.csv")
    if engine is None:
        sales = pd.read_csv(path.parent / "fact_daily_sales.csv", parse_dates=["date"])
        source_type = "committed_ingestion_csv"
    else:
        if engine.dialect.name != "postgresql":
            raise ValueError("The database source must be PostgreSQL.")
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
            with connection.begin():
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                sales = pd.read_sql_query(
                    text("""
                    SELECT store_id, item_id, date, sales, is_available
                    FROM fact_daily_sales ORDER BY store_id, item_id, date
                """),
                    connection,
                )
        source_type = "postgresql_snapshot"
    sales["unique_id"] = sales["store_id"] + "/" + sales["item_id"]
    if set(sales["unique_id"]) != set(selected["unique_id"]):
        raise ValueError("Sales series differ from the manifest-selected panel.")
    if len(sales) != manifest["sales_rows"]:
        raise ValueError("Sales row count differs from the committed ingestion scope.")
    history = validate_history(sales.rename(columns={"date": "ds", "sales": "y"}))
    if (
        str(history["ds"].min().date()) != manifest["sales_start"]
        or str(history["ds"].max().date()) != manifest["sales_end"]
    ):
        raise ValueError("Sales date range differs from the committed ingestion scope.")
    provenance = {
        "source_type": source_type,
        "ingestion_run_id": manifest["run_id"],
        "manifest_sha256": checksum(path),
        "selection_cutoff": manifest["config"]["selection_cutoff"],
        "ingestion_sales_sha256": manifest["artifact_sha256"]["fact_daily_sales.csv"],
        "bottom_series": int(history["unique_id"].nunique()),
        "snapshot_content_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(
                history[["unique_id", "ds", "y", "is_available"]],
                index=False,
            )
            .to_numpy()
            .tobytes()
        ).hexdigest(),
        "sales_rows": len(history),
        "sales_start": str(history["ds"].min().date()),
        "sales_end": str(history["ds"].max().date()),
    }
    return history, provenance
