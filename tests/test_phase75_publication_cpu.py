import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location(
        "phase75_publication", ROOT / "docs/_ext/wiki_phase75.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def payload(root, primary=False):
    m = module()
    source = root / m.SOURCE
    source.mkdir(parents=True)
    role = {
        "counts": {"events": 600},
        "by_category": {
            c: {"processed": 100}
            for c in ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar")
        },
        "by_retained_channel": {"sample": {"trials": 1}},
        "by_truth_fsp_size": {},
        "positive_cases": [],
    }
    value = {
        "status": "COMPLETED",
        "stage": "development",
        "primary_eligible": primary,
        "arms": {
            a: {
                "curves": {
                    "tiny": [{"loss": 1}] * 1000,
                    "downstream": [{"loss": 2}] * 1500,
                },
                "roles": {"heldout": role},
            }
            for a in ("joint", "project_conflicting_relation")
        },
    }
    data = json.dumps(value, separators=(",", ":")).encode()
    packed = gzip.compress(data, mtime=0)
    (source / "phase75-review.json.gz").write_bytes(packed)
    (source / "binding.json").write_text(
        json.dumps(
            {
                "file": "phase75-review.json.gz",
                "compressed_sha256": hashlib.sha256(packed).hexdigest(),
                "decoded_sha256": hashlib.sha256(data).hexdigest(),
                "decoded_bytes": len(data),
                "source_sha": "a" * 40,
            }
        )
    )
    out = root / "out"
    out.mkdir()
    (out / "phase74-decoder.txt").write_text("phase74 test decoder fixture")
    return m, out, data


def test_successor_delivery_preserves_curves_without_dashboard_duplication(tmp_path):
    m, out, data = payload(tmp_path)
    result = m.generate(tmp_path, out)
    compact = m.sibling("wiki_phase72_compact")
    assert (
        compact.decode(json.loads((out / "phase75-review.json").read_text()), 5_000_000)
        == data
    )
    assert "curves" not in result["arms"]["joint"]
    assert "by_retained_channel" not in result["arms"]["joint"]["roles"]["heldout"]


def test_successor_delivery_rejects_primary_claim(tmp_path):
    m, out, _ = payload(tmp_path, True)
    with pytest.raises(ValueError, match="scope"):
        m.generate(tmp_path, out)


def test_completed_successor_render_keeps_counts_limits_and_valid_headings(tmp_path):
    m = module()
    m.sibling("wiki_phase74").generate(ROOT, tmp_path)
    record = m.generate(ROOT, tmp_path)
    assert record["control_replay_training_history"] == {
        "tiny": True,
        "downstream": True,
    }
    for arm in record["arms"].values():
        assert (
            arm["sampling_accounting"]["downstream"]["actual_distinct_presented"]
            == 1535
        )
        assert arm["compute_accounting"]["total_encoder_passes"] == 44320
    lines = m.render(record)
    assert "Physical tree/pool/deep-survival" in "\n".join(lines)
    for contents in (lines, (ROOT / "docs/wiki/phase75.rst").read_text().splitlines()):
        for i, line in enumerate(contents):
            if i and line and set(line) <= set("=-~"):
                assert len(line) >= len(contents[i - 1]), contents[i - 1]
