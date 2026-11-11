"""Download, scope, validate, export and load M5 into PostgreSQL.

Run ``python -m supply_chain_opt_engine.ingestion --help``. Importing this module
does not download data or connect to a database.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from datetime import UTC, date, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from supply_chain_opt_engine.database import initialize_schema, load_dataset
from supply_chain_opt_engine.m5 import SOURCE_FILES, ScopeConfig, checksum, prepare_dataset
from supply_chain_opt_engine.metadata import ECONOMICS_PARAMETERS, generate_economics

LOGGER = logging.getLogger(__name__)


def download_m5(raw_dir: Path) -> None:
    """Download only required files using an authenticated Kaggle CLI."""
    executable = shutil.which("kaggle")
    if not executable:
        raise ValueError("Install the data extra with uv sync --extra data to use Kaggle download.")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        if (raw_dir / name).is_file():
            LOGGER.info("Keeping existing %s", name)
            continue
        with tempfile.TemporaryDirectory(prefix="m5-download-", dir=raw_dir) as temp:
            result = subprocess.run(
                [
                    executable,
                    "competitions",
                    "download",
                    "-c",
                    "m5-forecasting-accuracy",
                    "-f",
                    name,
                    "-p",
                    temp,
                    "-q",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                raise ValueError(
                    "Kaggle download failed. Configure Kaggle API credentials, accept the M5 "
                    "competition rules in your browser, and check network access."
                )
            source = Path(temp) / name
            if not source.exists():
                archive = Path(temp) / f"{name}.zip"
                if not archive.is_file():
                    raise ValueError(f"Kaggle did not return {name} or {name}.zip.")
                with zipfile.ZipFile(archive) as zipped:
                    with zipped.open(name) as incoming, source.open("wb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
            source.replace(raw_dir / name)
        LOGGER.info("Downloaded %s", name)


def read_config(path: Path, cutoff: date | None = None) -> tuple[dict, ScopeConfig, Path]:
    document = yaml.safe_load(path.read_text())
    data = document["data"]
    cutoff_value = cutoff or data.get("selection_cutoff")
    if not cutoff_value:
        raise ValueError(
            "Set data.selection_cutoff or pass --selection-cutoff YYYY-MM-DD. "
            "Choose a day no later than the earliest evaluation origin."
        )
    selection_cutoff = date.fromisoformat(str(cutoff_value))
    cutoffs = document.get("backtest", {}).get("cutoff_dates", [])
    if cutoffs and selection_cutoff > min(date.fromisoformat(str(value)) for value in cutoffs):
        raise ValueError("Selection cutoff cannot be later than the earliest backtest origin.")
    if data.get("selection") != "stratified_velocity_and_intermittency":
        raise ValueError("Unsupported series selection strategy.")
    config = ScopeConfig(
        selection_cutoff=selection_cutoff,
        stores=tuple(data["stores"]),
        categories=tuple(data["categories"]),
        state=data["state"],
        max_bottom_series=data["max_bottom_series"],
        history_days=data["history_days"],
        seed=document["seed"],
        chunk_size=data.get("chunk_size", 512),
    )
    return document, config, path.resolve().parent.parent


def run_ingestion(config_path: Path, cutoff: date | None, prepare_only: bool) -> Path:
    document, config, root = read_config(config_path, cutoff)
    dataset = prepare_dataset(root / document["data"]["raw_dir"], config)
    economics = generate_economics(dataset)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    interim = root / document["data"]["interim_dir"]
    synthetic = root / document["data"]["synthetic_dir"] / run_id
    interim.mkdir(parents=True, exist_ok=True)
    bundle = interim / run_id
    versions = {}
    for name in ["pandas", "numpy", "sqlalchemy", "psycopg", "psycopg2", "pyyaml"]:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    manifest = {
        "run_id": run_id,
        "status": "prepared",
        "config": config.manifest(),
        "source_sha256": dataset.source_checksums,
        "versions": versions,
        "economics_parameters": ECONOMICS_PARAMETERS,
        "sales_start": str(dataset.sales["date"].min().date()),
        "sales_end": str(dataset.sales["date"].max().date()),
        "bottom_series": len(dataset.selected),
        "distinct_items": len(economics),
        "calendar_rows": len(dataset.calendar),
        "sales_rows": len(dataset.sales),
        "price_rows": len(dataset.prices),
        "selection_summary": dataset.selected.groupby(["store_id", "cat_id", "demand_type"])
        .size()
        .rename("series")
        .reset_index()
        .to_dict(orient="records"),
        "assumptions": {
            "availability": "Weekly price exists; unlisted days retain observed zero sales.",
            "selection": "Training-listed demand only; intermittent means zero-share >= 0.5.",
            "economics": "Synthetic, seeded per item; training-week median price reference.",
            "holding_cost": "Unit purchase cost * annual carry rate 20% to 35% / 365.",
            "stockout_penalty": "3.1 to 7.9 times daily holding cost, rounded to 6 decimals.",
            "calendar_prices": "Full calendar and scoped price history, including future dates.",
        },
    }
    frames = {
        "selected_series.csv": dataset.selected,
        "dim_calendar.csv": dataset.calendar,
        "fact_daily_sales.csv": dataset.sales,
        "dim_prices.csv": dataset.prices,
        "dim_sku_economics.csv": economics,
    }
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=interim) as temporary:
        staging = Path(temporary)
        for name, frame in frames.items():
            frame.to_csv(staging / name, index=False, date_format="%Y-%m-%d")
        manifest["artifact_sha256"] = {name: checksum(staging / name) for name in frames}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(bundle)
    synthetic.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle / "dim_sku_economics.csv", synthetic / "dim_sku_economics.csv")
    LOGGER.info(
        "Prepared %d series and %d sales rows at %s",
        len(dataset.selected),
        len(dataset.sales),
        bundle,
    )
    if not prepare_only:
        load_dotenv(root / ".env", override=False)
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise ValueError(f"Set DATABASE_URL. Prepared artifacts remain at {bundle}.")
        engine = create_engine(url, connect_args={"connect_timeout": 10}, hide_parameters=True)
        try:
            initialize_schema(engine, root / "sql/01_schema.sql")
            manifest["loaded_rows"] = load_dataset(engine, dataset, economics)
        finally:
            engine.dispose()
        manifest["status"] = "loaded"
        manifest["loaded_at"] = datetime.now(UTC).isoformat()
        temporary_manifest = bundle / ".manifest.json.tmp"
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary_manifest.replace(bundle / "manifest.json")
        LOGGER.info("PostgreSQL transaction committed.")
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download", help="Download required M5 files via Kaggle")
    download.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ingest = subparsers.add_parser("load", help="Scope, validate, export and load M5")
    ingest.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ingest.add_argument("--selection-cutoff", type=date.fromisoformat)
    ingest.add_argument(
        "--prepare-only", action="store_true", help="Export without connecting to PostgreSQL"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.command == "download":
            download_m5(args.raw_dir)
        else:
            bundle = run_ingestion(args.config, args.selection_cutoff, args.prepare_only)
            print(bundle / "manifest.json")
    except SQLAlchemyError:
        LOGGER.error(
            "PostgreSQL load failed; transaction rolled back. Check database access and schema."
        )
        return 1
    except ImportError as error:
        LOGGER.error("Missing dependency %s. Run uv sync before loading PostgreSQL.", error.name)
        return 1
    except (ValueError, OSError, KeyError, zipfile.BadZipFile, yaml.YAMLError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
