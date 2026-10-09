"""Diagnostic-only lossless delivery, mutation rejection and scope preservation."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase77_publication", ROOT / "docs/_ext/wiki_phase77.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_complete_diagnostic_delivery(tmp_path):
    M.sibling("wiki_phase74").generate(ROOT, tmp_path)
    projected = M.generate(ROOT, tmp_path)
    decoded = M.sibling("wiki_phase72_compact").decode(
        json.loads((tmp_path / "phase77-review.json").read_text()), 5_000_000
    )
    source = gzip.decompress((ROOT / M.SOURCE / "phase77-review.json.gz").read_bytes())
    assert decoded == source
    v = json.loads(decoded)
    assert v["decision"]["scientific_training_jobs"] == []
    assert v["decision"]["fresh_development_designations"] == 0
    assert len(v["training_history"]["curves"]["downstream"]) == 1500
    assert len(v["training_history"]["curves"]["tiny"]) == 1000
    assert (
        v["historical_native_endpoints"]["tiny"]["counts"]["raw_exact_memberships"]
        == 32
    )
    assert (
        v["historical_native_endpoints"]["tiny"]["counts"]["accepted_exact_memberships"]
        == 27
    )
    assert v["populations"]["train"]["stages"]["proposal"]["B_to_unassigned"] == 4172
    assert set(v["populations"]["train"]["by_category"]) == {
        "charged",
        "mixed",
        "ccbar",
        "uubar",
        "ddbar",
        "ssbar",
    }
    assert (
        v["populations"]["train"]["stages"]["proposal"]["oracle_foreground_raw_exact"]
        == 0
    )
    assert "training_history" not in projected
    for lines in (
        M.render(projected),
        (ROOT / "docs/wiki/phase77.rst").read_text().splitlines(),
    ):
        for i, line in enumerate(lines):
            if i and line and set(line) <= set("=-~"):
                assert len(line) >= len(lines[i - 1]), lines[i - 1]


@pytest.mark.parametrize("mutation", ["hash", "scope", "population", "replay"])
def test_reject_changed_binding_or_scope(tmp_path, mutation):
    source = tmp_path / M.SOURCE
    source.parent.mkdir(parents=True)
    shutil.copytree(ROOT / M.SOURCE, source)
    p = source / "phase77-review.json.gz"
    if mutation == "hash":
        p.write_bytes(p.read_bytes() + b"corrupt")
    else:
        v = json.loads(gzip.decompress(p.read_bytes()))
        if mutation == "scope":
            v["decision"]["scientific_training_jobs"] = [1]
        if mutation == "population":
            v["populations"]["train"]["events"] = 600
        if mutation == "replay":
            v["populations"]["train"]["max_probability_difference"] = 0.1
        data = json.dumps(v).encode()
        packed = gzip.compress(data, mtime=0)
        p.write_bytes(packed)
        b = json.loads((source / "binding.json").read_text())
        b.update(
            compressed_sha256=hashlib.sha256(packed).hexdigest(),
            decoded_sha256=hashlib.sha256(data).hexdigest(),
            decoded_bytes=len(data),
        )
        (source / "binding.json").write_text(json.dumps(b))
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError):
        M.generate(tmp_path, out)
