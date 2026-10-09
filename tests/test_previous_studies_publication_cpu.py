"""Complete historical publication transport and fail-closed evidence binding."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("previous_studies", ROOT / "docs/_ext/wiki_previous_studies.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_lossless_full_inventory_and_decoder(tmp_path):
    result = M.generate(ROOT, tmp_path)
    binding = json.loads((ROOT / M.SOURCE / "binding.json").read_text())
    assert result["primary_eligible"] is False
    assert result["public_summary"]["phase83"]["qualifying_steps"] == 0
    assert len(result["metric_downloads"]) == len(binding["parts"])
    assert {p.name for p in tmp_path.iterdir()} == M.FILES
    for part in binding["parts"]:
        encoded = tmp_path / part["file"].removesuffix(".gz")
        actual = M.sibling("wiki_phase72_compact").decode(json.loads(encoded.read_text()), M.LIMIT)
        assert actual == gzip.decompress((ROOT / M.SOURCE / part["file"]).read_bytes())
    subprocess.run([sys.executable, "previous-studies-decoder.txt"], cwd=tmp_path, check=True)
    assert len(list(tmp_path.glob("*-decoded.json"))) == len(binding["parts"])
    for lines in (M.render(result),):
        for i, line in enumerate(lines):
            if i and line and set(line) <= set("=-~"):
                assert len(line) >= len(lines[i - 1])


@pytest.mark.parametrize("mutation", ["scope", "endpoint", "registry", "compressed", "inventory", "decoded", "source_coverage", "private_payload"])
def test_evidence_mutation_rejected(tmp_path, mutation):
    source = tmp_path / M.SOURCE
    source.parent.mkdir(parents=True)
    shutil.copytree(ROOT / M.SOURCE, source)
    binding = json.loads((source / "binding.json").read_text())
    if mutation == "scope": binding["fresh_validation_events"] = 1
    elif mutation == "endpoint": binding["public_summary"]["phase83"]["qualifying_steps"] = 1
    elif mutation == "registry": binding["parts"][0]["file"] = "../unsafe.json.gz"
    elif mutation == "compressed":
        p = source / binding["parts"][0]["file"]
        p.write_bytes(p.read_bytes() + b"broken")
    elif mutation == "private_payload":
        part = binding["parts"][0]
        p = source / part["file"]
        value = json.loads(gzip.decompress(p.read_bytes()))
        value["fragments"][0]["value"] = {"hostname": "compute999.internal"}
        data = M.canonical(value); packed = gzip.compress(data, mtime=0); p.write_bytes(packed)
        part.update(compressed_sha256=hashlib.sha256(packed).hexdigest(), decoded_sha256=hashlib.sha256(data).hexdigest(), decoded_bytes=len(data))
    elif mutation == "decoded": binding["parts"][0]["decoded_bytes"] += 1
    elif mutation == "inventory": (source / "inventory.json").write_text("{}")
    elif mutation == "source_coverage":
        p = source / "inventory.json"
        value = json.loads(p.read_text()); value["metric_sources"].pop()
        data = M.canonical(value); p.write_bytes(data)
        binding["inventory"]["sha256"] = hashlib.sha256(data).hexdigest()
    (source / "binding.json").write_text(json.dumps(binding))
    output = tmp_path / "output"; output.mkdir()
    with pytest.raises(ValueError): M.generate(tmp_path, output)


def test_fragment_reassembly_preserves_arrays_and_empty_objects():
    roots = {}
    M.insert_fragment(roots, {"source": "example", "path": ["rows", 0], "value": {}})
    M.insert_fragment(roots, {"source": "example", "path": ["rows", 1], "value": [0, False, None]})
    assert roots == {"example": {"rows": [{}, [0, False, None]]}}
    with pytest.raises(ValueError):
        M.insert_fragment(roots, {"source": "example", "path": ["rows", 1], "value": 9})


def test_shared_references_reject_cycles_and_invalid_indices():
    inventory = {"deduplication": {"format": "exact-shared-subtrees-v1", "reference_key": "$ref", "definitions_source": "shared-subtree-definitions", "definitions": 1}}
    for definitions in ([{"$ref": 0}], [{"$ref": 2}], [{"$ref": True}]):
        with pytest.raises(ValueError):
            M.expand_sources({"shared-subtree-definitions": definitions, "metric": {"$ref": 0}}, inventory)
    assert M.expand_sources({"shared-subtree-definitions": [{"v": [1, 2]}], "metric": {"$ref": 0}}, inventory) == {"metric": {"v": [1, 2]}}
