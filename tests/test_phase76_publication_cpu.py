import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location(
        "phase76_publication", ROOT / "docs/_ext/wiki_phase76.py"
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
            for a in ("partial_context_on", "partial_context_off")
        },
    }
    data = json.dumps(value, separators=(",", ":")).encode()
    packed = gzip.compress(data, mtime=0)
    (source / "phase76-review.json.gz").write_bytes(packed)
    (source / "binding.json").write_text(
        json.dumps(
            {
                "file": "phase76-review.json.gz",
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
        compact.decode(json.loads((out / "phase76-review.json").read_text()), 5_000_000)
        == data
    )
    assert "curves" not in result["arms"]["partial_context_on"]
    assert (
        "by_retained_channel"
        not in result["arms"]["partial_context_on"]["roles"]["heldout"]
    )


def test_successor_delivery_rejects_primary_claim(tmp_path):
    m, out, _ = payload(tmp_path, True)
    with pytest.raises(ValueError, match="scope"):
        m.generate(tmp_path, out)


def test_delivery_rejects_hash_changes(tmp_path):
    m, out, _ = payload(tmp_path)
    p = tmp_path / m.SOURCE / "phase76-review.json.gz"
    p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        m.generate(tmp_path, out)


def test_delivery_rejects_missing_category_even_with_total_600(tmp_path):
    m, out, _ = payload(tmp_path)
    p = tmp_path / m.SOURCE / "phase76-review.json.gz"
    value = json.loads(gzip.decompress(p.read_bytes()))
    del value["arms"]["partial_context_on"]["roles"]["heldout"]["by_category"][
        "charged"
    ]
    data = json.dumps(value).encode()
    packed = gzip.compress(data, mtime=0)
    p.write_bytes(packed)
    binding_path = p.parent / "binding.json"
    b = json.loads(binding_path.read_text())
    b.update(
        compressed_sha256=hashlib.sha256(packed).hexdigest(),
        decoded_sha256=hashlib.sha256(data).hexdigest(),
        decoded_bytes=len(data),
    )
    binding_path.write_text(json.dumps(b))
    with pytest.raises(ValueError, match="coverage"):
        m.generate(tmp_path, out)


def test_completed_phase76_counts_and_all_download_metrics(tmp_path):
    m = module()
    m.sibling("wiki_phase74").generate(ROOT, tmp_path)
    r = m.generate(ROOT, tmp_path)
    assert r["control_replay_model_states"] == {"tiny": True, "downstream": True}
    assert r["preregistered_endpoint"]["passed"] is False
    assert "diagnostics" not in r
    decoded = json.loads(
        m.sibling("wiki_phase72_compact").decode(
            json.loads((tmp_path / "phase76-review.json").read_text()), 5_000_000
        )
    )
    assert set(decoded["diagnostics"]) == {"exploratory_phase75", "fixed_phase76"}
    for arm, v in decoded["arms"].items():
        assert (
            len(v["curves"]["tiny"]) == 1000 and len(v["curves"]["downstream"]) == 1500
        )
        assert (
            v["roles"]["heldout"]["counts"]["known_accepted_failed_membership_trials"]
            == 400
        )
        assert (
            v["roles"]["heldout"]["relation_ignored_pairs"]["status"]
            == "MEASURED_AFTER_GENERATION"
        )
        assert (
            v["roles"]["heldout"]["relation_availability_by_category"]["ccbar"][
                "detector_relation"
            ]["status"]
            == "UNAVAILABLE_NO_WITHIN_B_SUPPORT"
        )
        assert v["convergence_diagnostics"]["downstream"]["convergence_proven"] is False
    assert (
        decoded["arms"]["partial_context_on"]["roles"]["heldout"]["counts"][
            "continuum_accepted_events"
        ]
        == 52
    )
    assert (
        decoded["arms"]["partial_context_off"]["roles"]["heldout"]["counts"][
            "continuum_accepted_events"
        ]
        == 46
    )
    for lines in (
        m.render(r),
        (ROOT / "docs/wiki/phase76.rst").read_text().splitlines(),
    ):
        for i, line in enumerate(lines):
            if i and line and set(line) <= set("=-~"):
                assert len(line) >= len(lines[i - 1]), lines[i - 1]
