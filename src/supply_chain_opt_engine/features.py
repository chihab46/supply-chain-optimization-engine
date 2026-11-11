"""Install SQL feature views and export an origin-specific forecast panel.

Usage: ``python -m supply_chain_opt_engine.features --origin 2016-05-22``.
Demand features are anchored to the origin; future labels are never exported.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import uuid
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from supply_chain_opt_engine.m5 import checksum

LOGGER = logging.getLogger(__name__)


def install_feature_views(engine: Engine, sql_path: Path) -> None:
    """Install or replace views and the function in one PostgreSQL transaction."""
    if engine.dialect.name != "postgresql":
        raise ValueError("Feature engineering requires PostgreSQL.")
    with engine.begin() as connection:
        cursor = connection.connection.driver_connection.cursor()
        try:
            cursor.execute(sql_path.read_text())
        finally:
            cursor.close()


def validate_panel(frame: pd.DataFrame, origin: date, horizon: int) -> None:
    """Reject incomplete, duplicated or incorrectly dated modeling panels."""
    if frame.empty:
        raise ValueError("The feature query returned no series.")
    if {"sales", "y", "target"} & set(frame.columns):
        raise ValueError("Feature panels must not contain realized target demand.")
    keys = ["unique_id", "forecast_origin", "ds"]
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Null or duplicate feature-panel keys.")
    target_dates = pd.to_datetime(frame["ds"])
    origin_dates = pd.to_datetime(frame["forecast_origin"])
    if not origin_dates.eq(pd.Timestamp(origin)).all():
        raise ValueError("Feature rows have inconsistent forecast origins.")
    if not frame["horizon"].eq((target_dates - pd.Timestamp(origin)).dt.days).all():
        raise ValueError("Target dates disagree with the horizon column.")
    if (
        not frame["horizon"].between(1, horizon).all()
        or not frame.groupby("unique_id").size().eq(horizon).all()
    ):
        raise ValueError("Every series needs the complete requested forecast horizon.")
    numeric = frame.select_dtypes(include="number").to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(numeric).any():
        raise ValueError("Infinite feature value; NULL is required for undefined ratios.")


def read_feature_panel(
    engine: Engine,
    origin: date | None = None,
    horizon: int = 28,
    known_future_prices: bool = False,
) -> tuple[pd.DataFrame, date, dict]:
    """Use one read-only repeatable snapshot for origin, features and source counts."""
    if engine.dialect.name != "postgresql":
        raise ValueError("Feature engineering requires PostgreSQL.")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 28:
        raise ValueError("Horizon must be an integer between 1 and 28.")
    if not isinstance(known_future_prices, bool):
        raise ValueError("known_future_prices must be a boolean.")
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
        with connection.begin():
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            connection.exec_driver_sql("SET LOCAL statement_timeout = '120s'")
            if origin is None:
                origin = connection.execute(
                    text("SELECT MAX(date) FROM fact_daily_sales")
                ).scalar_one()
            if origin is None:
                raise ValueError("Load sales before exporting features.")
            frame = pd.read_sql_query(
                text("""
                SELECT * FROM build_forecast_features(
                    CAST(:origin AS DATE), :horizon, :known_future_prices
                ) ORDER BY unique_id, ds
            """),
                connection,
                params={
                    "origin": origin,
                    "horizon": horizon,
                    "known_future_prices": known_future_prices,
                },
            )
            source = (
                connection.execute(
                    text("""
                SELECT COUNT(*) AS sales_rows, MIN(date) AS sales_start,
                    MAX(date) AS sales_end,
                    COUNT(DISTINCT (store_id, item_id)) AS bottom_series
                FROM fact_daily_sales
            """)
                )
                .mappings()
                .one()
            )
            snapshot = dict(source)
            snapshot["postgresql_version"] = connection.exec_driver_sql(
                "SHOW server_version"
            ).scalar_one()
    validate_panel(frame, origin, horizon)
    if frame["unique_id"].nunique() != snapshot["bottom_series"]:
        raise ValueError("A join dropped or added series relative to the sales panel.")
    return frame, origin, snapshot


def export_features(
    engine: Engine,
    sql_path: Path,
    output_dir: Path,
    origin: date | None = None,
    horizon: int = 28,
    known_future_prices: bool = False,
) -> Path:
    install_feature_views(engine, sql_path)
    frame, origin, snapshot = read_feature_panel(engine, origin, horizon, known_future_prices)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = output_dir / run_id
    manifest = {
        "run_id": run_id,
        "status": "exported",
        "forecast_origin": origin.isoformat(),
        "horizon_days": horizon,
        "bottom_series": int(frame["unique_id"].nunique()),
        "feature_rows": len(frame),
        "columns": list(frame.columns),
        "source_snapshot": snapshot,
        "feature_sql_sha256": checksum(sql_path),
        "price_policy": "known_future_prices"
        if known_future_prices
        else "carry_forward_from_origin",
        "demand_policy": "Origin-relative lags; rolling windows end on the origin, inclusive.",
        "availability_policy": "Unlisted days excluded from demand statistics; counts exported.",
        "warmup_policy": "29 complete observed days per series; undefined features remain NULL.",
        "price_history_policy": "52 calendar weeks including as-of week; observed prices only.",
        "holiday_policy": "National events in either slot; today included; unknown tail is NULL.",
        "null_counts": {name: int(count) for name, count in frame.isna().sum().items()},
        "versions": {name: version(name) for name in ["pandas", "numpy", "sqlalchemy", "pyarrow"]},
    }
    with tempfile.TemporaryDirectory(prefix=".features-", dir=output_dir) as temporary:
        staging = Path(temporary)
        frame.to_parquet(staging / "features.parquet", index=False)
        manifest["artifact_sha256"] = checksum(staging / "features.parquet")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        staging.rename(bundle)
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--origin", type=date.fromisoformat, help="Defaults to the latest observed day"
    )
    parser.add_argument("--horizon", type=int, help="Defaults to forecast.horizon_days, or 28")
    parser.add_argument(
        "--known-future-prices",
        action="store_true",
        help="Use future weekly prices as an explicitly known scenario",
    )
    parser.add_argument(
        "--install-only", action="store_true", help="Install SQL objects without exporting"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    engine = None
    try:
        document = yaml.safe_load(args.config.read_text())
        root = args.config.resolve().parent.parent
        load_dotenv(root / ".env", override=False)
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise ValueError("Set DATABASE_URL in .env before building SQL features.")
        engine = create_engine(url, connect_args={"connect_timeout": 10}, hide_parameters=True)
        sql_path = root / "sql/02_feature_view.sql"
        if args.install_only:
            install_feature_views(engine, sql_path)
            LOGGER.info("SQL feature views and build_forecast_features installed.")
        else:
            horizon = (
                args.horizon
                if args.horizon is not None
                else document.get("forecast", {}).get("horizon_days", 28)
            )
            bundle = export_features(
                engine,
                sql_path,
                root / document["data"]["processed_dir"] / "features",
                args.origin,
                horizon,
                args.known_future_prices,
            )
            LOGGER.info("Exported forecast feature panel to %s", bundle)
            print(bundle / "manifest.json")
    except SQLAlchemyError:
        LOGGER.error(
            "Feature build failed. Check database access, loaded data, calendar and 29-day history."
        )
        return 1
    except (ValueError, OSError, KeyError, ImportError, yaml.YAMLError) as error:
        LOGGER.error("%s", error)
        return 1
    finally:
        if engine is not None:
            engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
