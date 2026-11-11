from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

from supply_chain_opt_engine.ingestion import download_m5, main, read_config, run_ingestion
from supply_chain_opt_engine.m5 import SOURCE_FILES, checksum, prepare_dataset
from supply_chain_opt_engine.metadata import generate_economics


def test_scope_preserves_grid_categories_and_demand_types(m5_raw, scope):
    dataset = prepare_dataset(m5_raw, scope)
    assert len(dataset.selected) == 18
    assert len(dataset.sales) == 18 * 42
    assert len(dataset.calendar) == 56  # Known future calendar is retained.
    assert dataset.sales["date"].max() == pd.Timestamp("2020-02-11")
    assert dataset.sales.groupby(["store_id", "item_id"]).size().eq(42).all()
    assert not dataset.sales.duplicated(["store_id", "item_id", "date"]).any()
    assert set(dataset.selected["demand_type"]) == {"fast", "intermittent"}
    assert len(dataset.selected[["store_id", "cat_id"]].drop_duplicates()) == 9
    assert not dataset.selected["item_id"].str.endswith("004").any()
    assert dataset.source_checksums == {name: checksum(m5_raw / name) for name in SOURCE_FILES}


def test_selection_and_economics_cannot_use_future_sales_or_prices(m5_raw, scope):
    original = prepare_dataset(m5_raw, scope)
    path = m5_raw / "sales_train_evaluation.csv"
    sales = pd.read_csv(path)
    sales[[f"d_{day}" for day in range(29, 43)]] = 100000
    # Leave the late-introduced items' pre-cutoff observations unchanged.
    sales.to_csv(path, index=False)
    price_path = m5_raw / "sell_prices.csv"
    prices = pd.read_csv(price_path)
    prices.loc[prices["wm_yr_wk"] >= 104, "sell_price"] = 999.0
    prices.to_csv(price_path, index=False)
    changed = prepare_dataset(m5_raw, scope)
    assert_frame_equal(original.selected, changed.selected)
    assert_frame_equal(generate_economics(original), generate_economics(changed))


def test_selection_is_independent_of_csv_order_and_chunk_size(m5_raw, scope):
    original = prepare_dataset(m5_raw, scope)
    for filename in ["sales_train_evaluation.csv", "sell_prices.csv"]:
        path = m5_raw / filename
        pd.read_csv(path).sample(frac=1, random_state=9).to_csv(path, index=False)
    changed = prepare_dataset(m5_raw, replace(scope, chunk_size=31))
    assert_frame_equal(original.selected, changed.selected)
    assert_frame_equal(generate_economics(original), generate_economics(changed))


def test_scopes_final_days_from_sales_not_future_calendar(m5_raw, scope):
    dataset = prepare_dataset(
        m5_raw, replace(scope, history_days=35, selection_cutoff=date(2020, 2, 4))
    )
    assert dataset.sales["date"].min() == pd.Timestamp("2020-01-08")
    assert dataset.sales["date"].max() == pd.Timestamp("2020-02-11")
    assert len(dataset.sales) == 18 * 35


def test_unlisted_zero_sales_are_preserved_and_flagged(m5_raw, scope):
    path = m5_raw / "sell_prices.csv"
    prices = pd.read_csv(path)
    mask = prices["item_id"].eq("FOODS_1_000") & prices["store_id"].eq("CA_1")
    prices.loc[~(mask & prices["wm_yr_wk"].eq(100))].to_csv(path, index=False)
    path = m5_raw / "sales_train_evaluation.csv"
    sales = pd.read_csv(path)
    mask = sales["item_id"].eq("FOODS_1_000") & sales["store_id"].eq("CA_1")
    sales.loc[mask, [f"d_{day}" for day in range(1, 8)]] = 0
    sales.to_csv(path, index=False)
    dataset = prepare_dataset(m5_raw, scope)
    unlisted = dataset.sales.loc[~dataset.sales["is_available"]]
    assert len(unlisted) == 7
    assert unlisted["sales"].eq(0).all()
    series = dataset.selected.set_index("unique_id").loc["CA_1/FOODS_1_000"]
    assert series["training_available_days"] == 21
    assert series["zero_sales_share"] == 0


def test_positive_sales_without_prices_are_rejected(m5_raw, scope):
    path = m5_raw / "sell_prices.csv"
    prices = pd.read_csv(path)
    prices.loc[~prices["wm_yr_wk"].eq(105)].to_csv(path, index=False)
    with pytest.raises(ValueError, match="Positive sales without"):
        prepare_dataset(m5_raw, scope)


@pytest.mark.parametrize("invalid", [-1, 1.5, float("nan"), float("inf"), 2**32])
def test_invalid_sales_are_rejected_before_integer_conversion(m5_raw, scope, invalid):
    path = m5_raw / "sales_train_evaluation.csv"
    sales = pd.read_csv(path)
    sales["d_1"] = sales["d_1"].astype(float)
    sales.loc[0, "d_1"] = invalid
    sales.to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite integers"):
        prepare_dataset(m5_raw, scope)


@pytest.mark.parametrize(
    "filename", ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]
)
def test_duplicate_source_keys_are_rejected(m5_raw, scope, filename):
    path = m5_raw / filename
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate key"):
        prepare_dataset(m5_raw, scope)


def test_calendar_gap_and_short_history_are_rejected(m5_raw, scope):
    with pytest.raises(ValueError, match="shorter"):
        prepare_dataset(m5_raw, replace(scope, history_days=43))
    path = m5_raw / "calendar.csv"
    frame = pd.read_csv(path)
    frame.drop(index=3).to_csv(path, index=False)
    with pytest.raises(ValueError, match="complete daily grid"):
        prepare_dataset(m5_raw, scope)


def test_conflicting_item_metadata_is_rejected(m5_raw, scope):
    path = m5_raw / "sales_train_evaluation.csv"
    sales = pd.read_csv(path)
    sales.loc[0, "dept_id"] = "FOODS_2"
    sales.to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting dept_id"):
        prepare_dataset(m5_raw, scope)


def test_cutoff_requires_training_and_evaluation_data(m5_raw, scope):
    for cutoff in [date(2020, 1, 10), date(2020, 2, 11)]:
        with pytest.raises(ValueError, match="28 training days"):
            prepare_dataset(m5_raw, replace(scope, selection_cutoff=cutoff))


def test_economics_are_shared_reproducible_and_within_schema_bounds(m5_raw, scope):
    dataset = prepare_dataset(m5_raw, scope)
    economics = generate_economics(dataset)
    assert_frame_equal(economics, generate_economics(dataset))
    assert economics["item_id"].is_unique
    assert len(economics) < len(dataset.selected)
    assert economics["lead_time_days"].isin([2, 3, 5]).all()
    assert economics["minimum_order_quantity"].gt(0).all()
    ratios = economics["stockout_penalty"] / economics["daily_holding_cost"]
    assert ratios.between(3, 8).all()
    assert economics["unit_purchase_cost"].gt(0).all()
    assert economics["unit_volume_m3"].gt(0).all()
    changed = generate_economics(replace(dataset, config=replace(scope, seed=43)))
    assert not changed.equals(economics)


def fixture_config(m5_raw, scope):
    root = m5_raw.parent.parent
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    # This short ingestion fixture has no seasonal backtest schedule.
    config["backtest"]["cutoff_dates"] = []
    config["data"].update(
        history_days=scope.history_days,
        max_bottom_series=scope.max_bottom_series,
        chunk_size=scope.chunk_size,
    )
    path = root / "configs/default.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(config))
    return path


def test_prepare_only_exports_auditable_bundle_without_database(m5_raw, scope, monkeypatch):
    path = fixture_config(m5_raw, scope)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    bundle = run_ingestion(path, scope.selection_cutoff, prepare_only=True)
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["status"] == "prepared"
    assert manifest["sales_rows"] == scope.max_bottom_series * scope.history_days
    assert manifest["config"]["selection_cutoff"] == "2020-01-28"
    for name, digest in manifest["artifact_sha256"].items():
        assert checksum(bundle / name) == digest
    synthetic = path.parent.parent / "data/synthetic" / manifest["run_id"] / "dim_sku_economics.csv"
    assert synthetic.read_bytes() == (bundle / "dim_sku_economics.csv").read_bytes()
    assert not list(bundle.parent.glob(".prepare-*"))


def test_missing_data_and_invalid_cutoffs_fail_with_guidance(m5_raw, scope):
    path = fixture_config(m5_raw, scope)
    assert main(["load", "--config", str(path)]) == 1
    config = yaml.safe_load(path.read_text())
    config["backtest"]["cutoff_dates"] = ["2020-01-27"]
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="earliest backtest"):
        read_config(path, scope.selection_cutoff)
    (m5_raw / "calendar.csv").unlink()
    with pytest.raises(FileNotFoundError, match="download subcommand"):
        prepare_dataset(m5_raw, scope)


def test_download_extracts_only_requested_files_and_skips_existing(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / SOURCE_FILES[0]).write_text("existing")
    calls = []
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/kaggle")

    def fake_download(command, **kwargs):
        calls.append(command)
        name = command[command.index("-f") + 1]
        output = Path(command[command.index("-p") + 1])
        with zipfile.ZipFile(output / f"{name}.zip", "w") as archive:
            archive.writestr(name, "downloaded")
            archive.writestr("../../unexpected.txt", "must never be extracted")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_download)
    download_m5(raw)
    assert len(calls) == 2
    assert (raw / SOURCE_FILES[0]).read_text() == "existing"
    assert all((raw / name).exists() for name in SOURCE_FILES)
    assert not (tmp_path / "unexpected.txt").exists()


def test_download_authentication_failure_leaves_no_partial_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/kaggle")
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 1)
    )
    with pytest.raises(ValueError, match="accept the M5"):
        download_m5(tmp_path)
    assert not list(tmp_path.glob("*.csv"))
