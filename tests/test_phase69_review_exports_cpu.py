"""Native denominator and fail-closed publication contracts for the Phase69 review."""

import importlib.util, json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    s = importlib.util.spec_from_file_location(
        "wiki_phase69", ROOT / "docs/_ext/wiki_phase69.py"
    )
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_native_scalar_walk_retains_null_false_and_denominator():
    from scripts.export_phase69_native_metrics import leaves

    assert list(leaves({"n": 0, "d": 3, "v": None, "available": False})) == [
        {"metric": "available", "value": False},
        {"metric": "d", "value": 3},
        {"metric": "n", "value": 0},
        {"metric": "v", "value": None},
    ]
    with pytest.raises(ValueError):
        list(leaves({"bad": float("nan")}))


def summary():
    return json.loads(
        (ROOT / "artifacts/codex/phase69_publication_20261005/summary.json").read_text()
    )


def test_all_registered_phase69_views_publish(tmp_path):
    m = module()
    d = summary()
    out = m.generate(ROOT, tmp_path, d)
    assert {v["view"] for v in out["downloads"]} == set(m.VIEWS)
    assert out["metric_count"] == sum(f["metric_count"] for f in d["files"])
    text = (tmp_path / "phase69-review-metrics.json").read_text()
    assert (
        "/project/" not in text and "event_uid" not in text and "16818739" not in text
    )
    bundle_path = tmp_path / "phase69-all-aggregate-metrics.json"
    bundle = json.loads(bundle_path.read_text())
    assert bundle["version"] == "phase69-complete-aggregate-bundle-v2"
    names = [".".join(bundle["metric_name_segments"][i] for i in parts)
             for parts in bundle["metric_names"]]
    decoded = {(bundle["views"][view], bundle["arms"][arm], names[metric]): value
               for view, arm, metric, value in bundle["records"]}
    expected = {}
    for binding in out["downloads"]:
        view = json.loads((tmp_path / binding["filename"]).read_text())
        expected.update({(view["view"], view["arms"][arm], view["metric_names"][metric]): value
                         for arm, metric, value in view["records"]})
    assert decoded == expected
    assert len(decoded) == len(bundle["records"]) == 165915
    assert bundle_path.stat().st_size < 4 * 1024 * 1024


@pytest.mark.parametrize("change", ["missing_view", "path", "hash", "count"])
def test_public_export_rejects_changed_binding(tmp_path, change):
    d = summary()
    if change == "missing_view":
        d["files"].pop()
    if change == "path":
        d["files"][0]["source"] = "../../private"
    if change == "hash":
        d["files"][0]["sha256"] = "0" * 64
    if change == "count":
        d["files"][0]["metric_count"] += 1
    with pytest.raises(ValueError):
        module().generate(ROOT, tmp_path, d)


def test_unreviewed_nested_summary_text_cannot_publish(tmp_path):
    d = summary()
    d["arms"]["refined_encoder"]["private_unknown_field"] = "/home/private/checkpoint.pt"
    with pytest.raises(ValueError, match="summary schema"):
        module().generate(ROOT, tmp_path, d)


def test_aggregate_pid_confusion_uses_public_class_indices(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "docs/_ext"))
    from wiki_privacy import _contains_private_fields
    module().generate(ROOT, tmp_path, summary())
    value = json.loads((tmp_path / "phase69-review-metrics.json").read_text())
    assert not _contains_private_fields(value)
    row = value["arms"]["refined_encoder"]["retained_primary"]["full/greedy"]["leaf_pid_confusion"][0]
    assert "predicted_pid_class_index" in row
    assert "truth_pid_class_index" in row
    assert "predicted_token" not in row
