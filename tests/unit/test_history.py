from __future__ import annotations

import json

import pytest

from supply_chain_opt_engine.history import ingestion_manifest, load_history
from supply_chain_opt_engine.m5 import checksum, prepare_dataset


@pytest.fixture
def history_bundle(m5_raw, scope):
    root = m5_raw.parent.parent
    dataset = prepare_dataset(m5_raw, scope)
    folder = root / "data/interim/example"
    folder.mkdir(parents=True)
    dataset.selected.to_csv(folder / "selected_series.csv", index=False)
    dataset.sales.to_csv(folder / "fact_daily_sales.csv", index=False, date_format="%Y-%m-%d")
    manifest = {
        "status": "loaded",
        "run_id": "example",
        "config": scope.manifest(),
        "sales_rows": len(dataset.sales),
        "sales_start": str(dataset.sales.date.min().date()),
        "sales_end": str(dataset.sales.date.max().date()),
        "artifact_sha256": {
            name: checksum(folder / name)
            for name in ["selected_series.csv", "fact_daily_sales.csv"]
        },
    }
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest))
    return root, path


def test_history_source_verifies_checksums_and_preserves_selection_provenance(history_bundle):
    root, path = history_bundle
    history, source = load_history(root)
    assert len(history) == 18 * 42
    assert history.unique_id.nunique() == 18
    assert source["selection_cutoff"] == "2020-01-28"
    assert source["source_type"] == "committed_ingestion_csv"
    assert source["manifest_sha256"] == checksum(path)
    _, repeated_source = load_history(root)
    assert source["snapshot_content_sha256"] == repeated_source["snapshot_content_sha256"]


def test_tampered_or_uncommitted_sources_are_rejected(history_bundle):
    root, path = history_bundle
    manifest = json.loads(path.read_text())
    manifest["status"] = "prepared"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="committed"):
        ingestion_manifest(root, path)
    manifest["status"] = "loaded"
    path.write_text(json.dumps(manifest))
    csv = path.parent / "fact_daily_sales.csv"
    csv.write_text(csv.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_history(root)
