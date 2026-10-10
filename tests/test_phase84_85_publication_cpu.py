"""Complete historical publication transport and fail-closed evidence binding."""
import base64
import gzip
import lzma
import zlib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs/_ext"))
SPEC = importlib.util.spec_from_file_location("phase84_85", ROOT / "docs/_ext/wiki_phase84_85.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_narrative_text_wrap_preserves_privacy(tmp_path):
    """Sphinx must not wrap a prose slash into an absolute-path-shaped token."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "conf.py").write_text("project='fixture'\nmaster_doc='index'\n")
    (source / "index.rst").write_bytes((ROOT / "docs/wiki/phase84_85.rst").read_bytes())
    for target in ("studies", "previous_studies", "_generated/status/downloads"):
        page = source / (target + ".rst")
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(":orphan:\n\nFixture\n=======\n")
    output = tmp_path / "text"
    subprocess.run([sys.executable, "-m", "sphinx", "-W", "-b", "text", str(source),
                    str(output)], check=True, capture_output=True, text=True)
    privacy = M.sibling("wiki_privacy")
    assert not privacy.violations((output / "index.txt").read_text())


@pytest.fixture(scope="module")
def external_public(tmp_path_factory):
    output = tmp_path_factory.mktemp("phase78-external")
    M.sibling("wiki_phase74").generate(ROOT, output)
    M.sibling("wiki_phase78").generate(ROOT, output)
    return {name: (output / name).read_bytes() for name in (
        "phase78-review.json", *(f"phase78-events-{i}.json" for i in range(3)))}


def seed_external(output, external_public):
    for name, data in external_public.items(): (output / name).write_bytes(data)


def test_lossless_full_inventory_and_decoder(tmp_path, external_public):
    seed_external(tmp_path, external_public)
    result = M.generate(ROOT, tmp_path)
    binding = json.loads((ROOT / M.SOURCE / "binding.json").read_text())
    assert result["primary_eligible"] is False
    inventory_source = ROOT / M.SOURCE / "inventory.json.gz"
    assert not (ROOT / M.SOURCE / "inventory.json").exists()
    assert (tmp_path / "phase84-85-inventory.json").read_bytes() == gzip.decompress(inventory_source.read_bytes())
    assert result["public_summary"]["fresh_raw"] == [0,0]
    assert len(result["metric_downloads"]) == len(binding["parts"])
    assert {p.name for p in tmp_path.iterdir()} == M.FILES | set(external_public)
    for part in binding["parts"]:
        encoded = tmp_path / part["file"].removesuffix(".gz")
        actual = M.sibling("wiki_phase72_compact").decode(json.loads(encoded.read_text()), M.LIMIT)
        assert actual == gzip.decompress((ROOT / M.SOURCE / part["file"]).read_bytes())
    subprocess.run([sys.executable, "phase84-85-decoder.txt"], cwd=tmp_path, check=True)
    assert len(list(tmp_path.glob("*-decoded.json"))) == len(binding["parts"])
    inventory = json.loads((tmp_path / "phase84-85-inventory.json").read_text())
    assert len(inventory["metric_sources"]) == 31
    histories = [r for r in inventory["metric_sources"] if r["source"].endswith("downstream-metrics.jsonl")]
    assert len(histories) == 6
    assert sum(r["public_cardinality"]["array_items"] for r in histories) == 36000
    decoded = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("phase84-85-source-*.json"))]
    fresh = next(v for v in decoded if isinstance(v, dict) and v.get("heldout_events") == 600 and "paired" in v)
    for arm, continuum in (("connection_off",48),("connection_on",58)):
        counts = fresh["arms"][arm]["aggregate"]["counts"]
        assert counts["raw_exact_memberships"] == counts["accepted_exact_memberships"] == 0
        assert counts["continuum_accepted_events"] == continuum
    assert not fresh["primary_coverage_met"]

    for lines in (M.render(result),):
        for i, line in enumerate(lines):
            if i and line and set(line) <= set("=-~"):
                assert len(line) >= len(lines[i - 1])


@pytest.mark.parametrize("mutation", ["scope", "endpoint", "registry", "compressed", "inventory", "decoded", "source_coverage", "private_payload", "external_hash", "external_count"])
def test_evidence_mutation_rejected(tmp_path, mutation, external_public):
    source = tmp_path / M.SOURCE
    source.parent.mkdir(parents=True)
    shutil.copytree(ROOT / M.SOURCE, source)
    binding = json.loads((source / "binding.json").read_text())
    if mutation == "external_hash": binding["external_sources"][0]["envelope_sha256"] = "0" * 64
    elif mutation == "external_count": binding["external_sources"].pop()
    elif mutation == "scope": binding["fresh_validation_events"] = 1
    elif mutation == "endpoint": binding["public_summary"]["fresh_raw"] = [1,0]
    elif mutation == "registry": binding["parts"][0]["file"] = "../unsafe.json.gz"
    elif mutation == "compressed":
        p = source / binding["parts"][0]["file"]
        p.write_bytes(p.read_bytes() + b"broken")
    elif mutation == "private_payload":
        part = binding["parts"][0]
        p = source / part["file"]
        wrapper = json.loads(gzip.decompress(p.read_bytes()))
        value = json.loads(M.decode_fragments(wrapper))
        value["fragments"][0]["value"] = {"hostname": "compute999.internal"}
        original = M.canonical(value)
        wrapper = inner_wrapper(original)
        part.update(fragment_decoded_bytes=len(original), fragment_decoded_sha256=hashlib.sha256(original).hexdigest())
        data = M.canonical(wrapper); packed = gzip.compress(data, mtime=0); p.write_bytes(packed)
        part.update(compressed_sha256=hashlib.sha256(packed).hexdigest(), decoded_sha256=hashlib.sha256(data).hexdigest(), decoded_bytes=len(data))
    elif mutation == "decoded": binding["parts"][0]["decoded_bytes"] += 1
    elif mutation == "inventory": (source / "inventory.json.gz").write_bytes(b"broken")
    elif mutation == "source_coverage":
        p = source / "inventory.json.gz"
        value = json.loads(gzip.decompress(p.read_bytes())); value["metric_sources"].pop()
        data = M.canonical(value); packed = gzip.compress(data, mtime=0); p.write_bytes(packed)
        binding["inventory"].update(sha256=hashlib.sha256(data).hexdigest(), compressed_sha256=hashlib.sha256(packed).hexdigest(), decoded_bytes=len(data))
    (source / "binding.json").write_text(json.dumps(binding))
    output = tmp_path / "output"; output.mkdir()
    seed_external(output, external_public)
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


@pytest.mark.parametrize("mutation", ["hash", "path", "source", "index"])
def test_external_subtree_reference_rejects_mutations(mutation):
    data = {"metric": [1, 2]}
    reference = {"file": "phase78-review.json", "path": ["metric"], "sha256": M.digest(M.canonical([1, 2]))}
    reference_index = 0
    if mutation == "hash": reference["sha256"] = "0" * 64
    elif mutation == "path": reference["path"] = ["missing"]
    elif mutation == "source": reference["file"] = "unregistered.json"
    elif mutation == "index": reference_index = 2
    roots = {"shared-subtree-definitions": [], "external-subtree-definitions": [reference], "metric": {"$external": reference_index}}
    inventory = {"deduplication": {"format": "exact-shared-subtrees-v1", "reference_key": "$ref", "definitions_source": "shared-subtree-definitions", "definitions": 0}}
    inventory["external_reference_count"] = 1
    with pytest.raises(ValueError): M.expand_sources(roots, inventory, {"phase78-review.json": data})


def test_external_download_bytes_remain_unchanged(tmp_path, external_public):
    seed_external(tmp_path, external_public)
    M.generate(ROOT, tmp_path)
    for name, data in external_public.items(): assert (tmp_path / name).read_bytes() == data


@pytest.mark.parametrize("mutation", ["bytes", "decoded_hash", "decoded_bytes"])
def test_external_envelopes_are_authenticated(tmp_path, external_public, mutation):
    seed_external(tmp_path, external_public)
    binding = json.loads((ROOT / M.SOURCE / "binding.json").read_text())
    records = binding["external_sources"]
    if mutation == "bytes":
        p = tmp_path / records[0]["filename"]
        p.write_bytes(p.read_bytes() + b"changed")
    elif mutation == "decoded_hash": records[0]["decoded_sha256"] = "0" * 64
    else: records[0]["decoded_bytes"] += 1
    with pytest.raises(ValueError): M.load_external_sources(tmp_path, records)


def inner_wrapper(data, packed=None):
    if packed is None: packed = lzma.compress(data, format=lzma.FORMAT_XZ, preset=6, check=lzma.CHECK_CRC64)
    return {"encoding": "bounded-xz-base32-fragments-v1", "decoded_bytes": len(data), "decoded_sha256": hashlib.sha256(data).hexdigest(), "data": base64.b32encode(packed).decode()}


@pytest.mark.parametrize("mutation", ["bomb", "trailing", "concatenated", "size", "hash", "memory", "corruption", "checksum", "limit"])
def test_inner_xz_bounds_and_integrity(mutation):
    data = b"{}"
    packed = lzma.compress(data, format=lzma.FORMAT_XZ, preset=6, check=lzma.CHECK_CRC64)
    value = inner_wrapper(data, packed)
    if mutation == "bomb": value = inner_wrapper(data, lzma.compress(b" " * 100000, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64))
    elif mutation == "trailing": value = inner_wrapper(data, packed + b"trailing")
    elif mutation == "concatenated": value = inner_wrapper(data, packed + packed)
    elif mutation == "size": value["decoded_bytes"] += 1
    elif mutation == "hash": value["decoded_sha256"] = "0" * 64
    elif mutation == "memory":
        # Change the tiny stream's LZMA2 dictionary request to64MiB, exceeding
        # the64MiB decoder allowance once required state is included.
        altered = bytearray(packed)
        assert altered[12:16] == bytes([2, 0, 33, 1])
        altered[16] = 28
        altered[20:24] = zlib.crc32(altered[12:20]).to_bytes(4, "little")
        value = inner_wrapper(data, bytes(altered))
    elif mutation == "corruption": value = inner_wrapper(data, packed[:30] + bytes([packed[30] ^ 255]) + packed[31:])
    elif mutation == "checksum": value = inner_wrapper(data, lzma.compress(data, format=lzma.FORMAT_XZ, check=lzma.CHECK_NONE))
    elif mutation == "limit": value["decoded_bytes"] = 5000001
    with pytest.raises(ValueError): M.decode_fragments(value)


def test_inner_xz_exact_roundtrip():
    data = M.canonical({"values": [0, -1, 0.123456789, None, False], "empty": []})
    assert M.decode_fragments(inner_wrapper(data)) == data


def test_complete_artifact_privacy_against_repository_sources(tmp_path, external_public):
    seed_external(tmp_path, external_public)
    M.generate(ROOT, tmp_path)
    # Match the production generated-directory validator; raw-source-body
    # hashes remain mandatory in this mode.
    report = M.sibling("wiki_privacy").validate_artifact(tmp_path, root=ROOT, generated_projection=True)
    assert report["privacy"] == "PASS"
    assert report["raw_source_hash_rules"] > 0


@pytest.mark.parametrize("mutation", ["trailing", "decoded_size", "decoded_hash"])
def test_inventory_archive_bounds(tmp_path, external_public, mutation):
    source = tmp_path / M.SOURCE; source.parent.mkdir(parents=True)
    shutil.copytree(ROOT / M.SOURCE, source)
    binding = json.loads((source / "binding.json").read_text())
    if mutation == "trailing":
        archive = source / "inventory.json.gz"
        packed = archive.read_bytes() + b"trailing"; archive.write_bytes(packed)
        binding["inventory"]["compressed_sha256"] = hashlib.sha256(packed).hexdigest()
    elif mutation == "decoded_size": binding["inventory"]["decoded_bytes"] += 1
    else: binding["inventory"]["sha256"] = "0" * 64
    (source / "binding.json").write_bytes(M.canonical(binding))
    output = tmp_path / "output"; output.mkdir(); seed_external(output, external_public)
    with pytest.raises(ValueError): M.generate(tmp_path, output)
