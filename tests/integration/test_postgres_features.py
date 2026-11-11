from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from supply_chain_opt_engine.database import load_dataset
from supply_chain_opt_engine.features import export_features, read_feature_panel
from supply_chain_opt_engine.m5 import checksum, prepare_dataset
from supply_chain_opt_engine.metadata import generate_economics

pytestmark = pytest.mark.integration


def test_feature_query_and_parquet_export(engine, m5_raw, scope, tmp_path):
    dataset = prepare_dataset(m5_raw, scope)
    load_dataset(engine, dataset, generate_economics(dataset))
    origin = date(2020, 1, 29)
    bundle = export_features(
        engine, Path("sql/02_feature_view.sql"), tmp_path / "exports", origin=origin, horizon=7
    )
    manifest = json.loads((bundle / "manifest.json").read_text())
    exported = pd.read_parquet(bundle / "features.parquet")
    assert manifest["feature_rows"] == len(exported) == 18 * 7
    assert manifest["price_policy"] == "carry_forward_from_origin"
    assert manifest["artifact_sha256"] == checksum(bundle / "features.parquet")
    assert manifest["source_snapshot"]["sales_rows"] == len(dataset.sales)
    assert not {"y", "sales", "target"} & set(exported.columns)
    assert not list(bundle.parent.glob(".features-*"))
    # Latest-origin default is independent of the origin used in the export.
    latest, actual_origin, _ = read_feature_panel(engine, horizon=14)
    assert actual_origin == date(2020, 2, 11)
    assert len(latest) == 18 * 14
