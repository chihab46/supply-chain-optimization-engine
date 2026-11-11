"""The PostgreSQL feature snapshot and committed extract must encode identically."""

from __future__ import annotations

import json
from datetime import date

import pytest
from pandas.testing import assert_frame_equal

from supply_chain_opt_engine.database import load_dataset
from supply_chain_opt_engine.feature_store import FeatureEncoder, load_feature_store
from supply_chain_opt_engine.m5 import checksum, prepare_dataset
from supply_chain_opt_engine.metadata import generate_economics

pytestmark = pytest.mark.integration


def test_live_snapshot_encodes_like_committed_ingestion(engine, m5_raw, scope):
    dataset = prepare_dataset(m5_raw, scope)
    economics = generate_economics(dataset)
    load_dataset(engine, dataset, economics)
    root = m5_raw.parent.parent
    folder = root / "data/interim/feature-source-test"
    folder.mkdir(parents=True)
    frames = {
        "selected_series.csv": dataset.selected,
        "fact_daily_sales.csv": dataset.sales,
        "dim_calendar.csv": dataset.calendar,
        "dim_prices.csv": dataset.prices,
        "dim_sku_economics.csv": economics,
    }
    for name, frame in frames.items():
        frame.to_csv(folder / name, index=False, date_format="%Y-%m-%d")
    manifest = {
        "status": "loaded",
        "run_id": "feature-source-test",
        "config": scope.manifest(),
        "sales_rows": len(dataset.sales),
        "sales_start": str(dataset.sales.date.min().date()),
        "sales_end": str(dataset.sales.date.max().date()),
        "artifact_sha256": {name: checksum(folder / name) for name in frames},
    }
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest))
    offline = load_feature_store(root, manifest_path=path)
    live = load_feature_store(root, engine=engine, manifest_path=path)
    encoder = FeatureEncoder.from_store(offline)
    assert_frame_equal(
        encoder.transform(offline.panel(date(2020, 1, 29), 7)),
        encoder.transform(live.panel(date(2020, 1, 29), 7)),
    )
