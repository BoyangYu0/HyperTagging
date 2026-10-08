import importlib.util
import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location(
        "phase74_review", ROOT / "docs/_ext/wiki_phase74.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_review_payload_preserves_counts_histories_and_privacy(tmp_path):
    m = module()
    out = tmp_path / "out"
    out.mkdir()
    v = m.generate(ROOT, out)
    assert v["coverage"]["heldout_distinct_events"] == 600 and not v["primary_eligible"]
    assert (
        v["arms"]["256-assembly"]["roles"]["heldout"]["counts"][
            "accepted_exact_memberships"
        ]
        == 1
    )
    compact = m.sibling("wiki_phase72_compact")
    data = compact.decode(
        json.loads((out / "phase74-review.json").read_text()), 5_000_000
    )
    decoded = json.loads(data)
    assert (
        sum(
            len(rows)
            for r in decoded["arms"].values()
            for rows in r["histories"].values()
        )
        == 14000
    )
    assert b"event_uid" not in data and b"/project/" not in data
    binding = json.loads((ROOT / m.SOURCE / "binding.json").read_text())
    assert hashlib.sha256(data).hexdigest() == binding["decoded_sha256"]
    assert "Physical exact-tree" in "\n".join(m.render(v))


def test_review_rejects_hash_change(tmp_path):
    m = module()
    source = tmp_path / m.SOURCE
    source.mkdir(parents=True)
    binding = json.loads((ROOT / m.SOURCE / "binding.json").read_text())
    (source / "binding.json").write_text(json.dumps(binding))
    (source / binding["file"]).write_bytes(b"corrupted")
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="hash mismatch"):
        m.generate(tmp_path, out)
