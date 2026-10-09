"""Lossless, authenticated publication of the frozen Phase79–83 evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import zlib

SOURCE = "artifacts/codex/previous_studies_review_20261009"
LIMIT = 5_000_000


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def part_names(binding):
    parts = binding["parts"]
    if not 1 <= len(parts) <= 40:
        raise ValueError("Previous-studies part count out of bounds")
    names = [p["file"] for p in parts]
    if names != [f"previous-studies-metrics-{i:03d}.json.gz" for i in range(len(parts))]:
        raise ValueError("Previous-studies part registry changed")
    return names


def files():
    binding_path = Path(__file__).resolve().parents[2] / SOURCE / "binding.json"
    names = part_names(json.loads(binding_path.read_text())) if binding_path.exists() else []
    return {n.removesuffix(".gz") for n in names} | {
        "previous-studies-integrity.json", "previous-studies-decoder.txt", "previous-studies-inventory.json"
    }


FILES = files()


def cardinality(value):
    from collections import Counter
    counts = Counter()
    def walk(item):
        if isinstance(item, dict):
            counts["objects"] += 1
            counts["object_fields"] += len(item)
            for child in item.values(): walk(child)
        elif isinstance(item, list):
            counts["arrays"] += 1
            counts["array_items"] += len(item)
            for child in item: walk(child)
        else:
            counts["scalars"] += 1
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                counts["numeric_scalars"] += 1
    walk(value)
    return dict(counts)


def insert_fragment(roots, fragment):
    source, path, value = fragment["source"], fragment["path"], fragment["value"]
    if not path:
        if source in roots: raise ValueError("Overlapping previous-studies fragment")
        roots[source] = value
        return
    if source not in roots: roots[source] = [] if type(path[0]) is int else {}
    node = roots[source]
    for index, key in enumerate(path):
        last = index == len(path) - 1
        if isinstance(node, list):
            if type(key) is not int or key < 0 or key > len(node):
                raise ValueError("Noncontiguous previous-studies array")
            if key == len(node): node.append(None)
        elif not isinstance(node, dict) or not isinstance(key, str):
            raise ValueError("Invalid previous-studies fragment path")
        if last:
            if (isinstance(node, dict) and key in node) or (isinstance(node, list) and node[key] is not None):
                raise ValueError("Overlapping previous-studies fragment")
            node[key] = value
        else:
            if (isinstance(node, dict) and key not in node) or (isinstance(node, list) and node[key] is None):
                node[key] = [] if type(path[index + 1]) is int else {}
            node = node[key]


def expand_sources(roots, inventory):
    policy = inventory["deduplication"]
    if policy["format"] != "exact-shared-subtrees-v1" or policy["reference_key"] != "$ref":
        raise ValueError("Unknown previous-studies shared-subtree format")
    definitions = roots.pop(policy["definitions_source"])
    if not isinstance(definitions, list) or len(definitions) != policy["definitions"]:
        raise ValueError("Previous-studies definition cardinality mismatch")
    active, cache = set(), {}
    def expand(value):
        if isinstance(value, dict):
            if set(value) == {"$ref"}:
                index = value["$ref"]
                if type(index) is not int or not 0 <= index < len(definitions) or index in active:
                    raise ValueError("Invalid or cyclic previous-studies reference")
                if index not in cache:
                    active.add(index)
                    cache[index] = expand(definitions[index])
                    active.remove(index)
                return cache[index]
            return {key: expand(child) for key, child in value.items()}
        if isinstance(value, list): return [expand(child) for child in value]
        return value
    return {key: expand(value) for key, value in roots.items()}


def generate(root, output):
    source = Path(root) / SOURCE
    if not source.exists():
        return None
    binding_path = source / "binding.json"
    if binding_path.is_symlink() or binding_path.stat().st_size > LIMIT:
        raise ValueError("Unsafe previous-studies binding")
    binding = json.loads(binding_path.read_text())
    if (binding["version"] != "previous-studies-public-binding-v1"
            or binding["source_role"] != "training"
            or binding["fresh_validation_events"] != 0
            or binding["primary_eligible"] is not False
            or binding["first_phase"] != 79 or binding["last_phase"] != 83
            or not re.fullmatch(r"[0-9a-f]{40}", binding["source_sha"])):
        raise ValueError("Previous-studies scope changed")
    summary = binding["public_summary"]
    if (summary["phase83"]["qualifying_steps"] != 0 or summary["phase83"]["steps"] != 12
            or summary["phase83"]["scientific_benefit"] is not False
            or summary["phase81"]["b_trials"] != 1024
            or any(summary["phase81"][key] != 0 for key in (
                "native_raw_exact", "pair_raw_exact", "native_accepted_exact", "pair_accepted_exact"))):
        raise ValueError("Previous-studies scientific endpoint changed")
    names = part_names(binding)
    codec = sibling("wiki_phase72_compact")
    privacy = sibling("wiki_privacy")
    if binding["inventory"]["file"] != "inventory.json":
        raise ValueError("Unsafe previous-studies inventory name")
    inventory_path = source / "inventory.json"
    if inventory_path.is_symlink() or inventory_path.stat().st_size > LIMIT:
        raise ValueError("Unsafe previous-studies inventory")
    inventory_bytes = inventory_path.read_bytes()
    if digest(inventory_bytes) != binding["inventory"]["sha256"]:
        raise ValueError("Previous-studies inventory hash mismatch")
    inventory = privacy._strict_json(inventory_bytes.decode())
    if privacy._contains_private_fields(inventory) or privacy.redact(inventory_bytes.decode()) != inventory_bytes.decode():
        raise ValueError("Private previous-studies inventory")
    records, fragment_ids, reconstructed = [], set(), {}
    for name, part in zip(names, binding["parts"]):
        path = source / name
        if path.is_symlink() or path.stat().st_size > LIMIT:
            raise ValueError("Unsafe previous-studies part")
        packed = path.read_bytes()
        if digest(packed) != part["compressed_sha256"]:
            raise ValueError("Previous-studies compressed hash mismatch")
        decoder = zlib.decompressobj(31)
        data = decoder.decompress(packed, LIMIT + 1)
        if (len(data) > LIMIT or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data
                or len(data) != part["decoded_bytes"] or digest(data) != part["decoded_sha256"]):
            raise ValueError("Previous-studies decoded hash or bounds mismatch")
        value = privacy._strict_json(data.decode())
        if value["version"] != "previous-studies-lossless-fragments-v1":
            raise ValueError("Wrong previous-studies fragment version")
        if len(value["fragments"]) != part["fragments"]:
            raise ValueError("Previous-studies fragment cardinality mismatch")
        for fragment in value["fragments"]:
            key = (fragment["source"], canonical(fragment["path"]))
            if key in fragment_ids:
                raise ValueError("Duplicate previous-studies fragment")
            fragment_ids.add(key)
            insert_fragment(reconstructed, fragment)
        envelope = codec.encode(data, packed, LIMIT)
        filename = name.removesuffix(".gz")
        (output / filename).write_bytes(envelope)
        records.append({"filename": filename, "sha256": digest(envelope), "bytes": len(envelope),
                        "decoded_sha256": digest(data), "decoded_bytes": len(data),
                        "fragments": len(value["fragments"])})
    reconstructed = expand_sources(reconstructed, inventory)
    expected_sources = {r["source"] for r in inventory["metric_sources"]}
    if len(expected_sources) != len(inventory["metric_sources"]) or set(reconstructed) != expected_sources:
        raise ValueError("Previous-studies source coverage mismatch")
    for record in inventory["metric_sources"]:
        value = reconstructed[record["source"]]
        data = canonical(value)
        if (digest(data) != record["sanitized_sha256"] or len(data) != record["sanitized_bytes"]
                or cardinality(value) != record["public_cardinality"]):
            raise ValueError("Previous-studies reconstruction or cardinality mismatch")
    (output / "previous-studies-inventory.json").write_bytes(inventory_bytes)
    integrity = {"version": "previous-studies-integrity-v1", "source_sha": binding["source_sha"],
                 "source_role": "training", "fresh_validation_events": 0,
                 "inventory_sha256": digest(inventory_bytes), "parts": records,
                 "fragment_count": len(fragment_ids), "all_supported_metrics_retained": True}
    (output / "previous-studies-integrity.json").write_bytes(canonical(integrity))
    (output / "previous-studies-decoder.txt").write_text(DECODER)
    return {"status": "COMPLETED", "stage": "training_only_sequential_evidence", "primary_eligible": False,
            "source_sha": binding["source_sha"], "public_summary": binding["public_summary"],
            "metric_downloads": records, "fragment_count": len(fragment_ids)}


DECODER = '''# Python 3: verify and decode every complete metric fragment envelope.
import base64, hashlib, json, pathlib, zlib
integrity = json.loads(pathlib.Path("previous-studies-integrity.json").read_text())
for part in integrity["parts"]:
    raw = pathlib.Path(part["filename"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == part["sha256"]
    e = json.loads(raw)
    assert set(e) == {"encoding", "decoded_bytes", "decoded_sha256", "data"}
    assert e["encoding"] == "bounded-gzip-base32-json-v1"
    assert type(e["decoded_bytes"]) is int and 0 <= e["decoded_bytes"] <= 5000000
    assert len(e["data"]) <= 10485760
    d = zlib.decompressobj(31)
    value = d.decompress(base64.b32decode(e["data"], casefold=False), e["decoded_bytes"] + 1)
    assert d.eof and not d.unused_data and not d.unconsumed_tail
    assert len(value) == e["decoded_bytes"]
    assert hashlib.sha256(value).hexdigest() == e["decoded_sha256"] == part["decoded_sha256"]
    pathlib.Path(part["filename"].replace(".json", "-decoded.json")).write_bytes(value)
# Every fragment can be rebuilt into the exact privacy-safe source below.
'''

DECODER += r'''
def cardinality(value):
    from collections import Counter
    counts = Counter()
    def walk(item):
        if isinstance(item, dict):
            counts["objects"] += 1
            counts["object_fields"] += len(item)
            for child in item.values(): walk(child)
        elif isinstance(item, list):
            counts["arrays"] += 1
            counts["array_items"] += len(item)
            for child in item: walk(child)
        else:
            counts["scalars"] += 1
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                counts["numeric_scalars"] += 1
    walk(value)
    return dict(counts)


def insert_fragment(roots, fragment):
    source, path, value = fragment["source"], fragment["path"], fragment["value"]
    if not path:
        if source in roots: raise ValueError("Overlapping previous-studies fragment")
        roots[source] = value
        return
    if source not in roots: roots[source] = [] if type(path[0]) is int else {}
    node = roots[source]
    for index, key in enumerate(path):
        last = index == len(path) - 1
        if isinstance(node, list):
            if type(key) is not int or key < 0 or key > len(node):
                raise ValueError("Noncontiguous previous-studies array")
            if key == len(node): node.append(None)
        elif not isinstance(node, dict) or not isinstance(key, str):
            raise ValueError("Invalid previous-studies fragment path")
        if last:
            if (isinstance(node, dict) and key in node) or (isinstance(node, list) and node[key] is not None):
                raise ValueError("Overlapping previous-studies fragment")
            node[key] = value
        else:
            if (isinstance(node, dict) and key not in node) or (isinstance(node, list) and node[key] is None):
                node[key] = [] if type(path[index + 1]) is int else {}
            node = node[key]


def expand_sources(roots, inventory):
    policy = inventory["deduplication"]
    if policy["format"] != "exact-shared-subtrees-v1" or policy["reference_key"] != "$ref":
        raise ValueError("Unknown previous-studies shared-subtree format")
    definitions = roots.pop(policy["definitions_source"])
    if not isinstance(definitions, list) or len(definitions) != policy["definitions"]:
        raise ValueError("Previous-studies definition cardinality mismatch")
    active, cache = set(), {}
    def expand(value):
        if isinstance(value, dict):
            if set(value) == {"$ref"}:
                index = value["$ref"]
                if type(index) is not int or not 0 <= index < len(definitions) or index in active:
                    raise ValueError("Invalid or cyclic previous-studies reference")
                if index not in cache:
                    active.add(index)
                    cache[index] = expand(definitions[index])
                    active.remove(index)
                return cache[index]
            return {key: expand(child) for key, child in value.items()}
        if isinstance(value, list): return [expand(child) for child in value]
        return value
    return {key: expand(value) for key, value in roots.items()}



inventory_raw = pathlib.Path("previous-studies-inventory.json").read_bytes()
assert hashlib.sha256(inventory_raw).hexdigest() == integrity["inventory_sha256"]
inventory = json.loads(inventory_raw)
roots = {}
for part in integrity["parts"]:
    decoded = pathlib.Path(part["filename"].replace(".json", "-decoded.json"))
    for fragment in json.loads(decoded.read_bytes())["fragments"]:
        insert_fragment(roots, fragment)
roots = expand_sources(roots, inventory)
assert set(roots) == {r["source"] for r in inventory["metric_sources"]}
for i, record in enumerate(inventory["metric_sources"]):
    raw = (json.dumps(roots[record["source"]], sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    assert cardinality(roots[record["source"]]) == record["public_cardinality"]
    assert len(raw) == record["sanitized_bytes"]
    assert hashlib.sha256(raw).hexdigest() == record["sanitized_sha256"]
    pathlib.Path("previous-studies-source-" + str(i).zfill(3) + ".json").write_bytes(raw)
'''


def render(value):
    lines = ["Phase83 completed optimizer-history diagnostic", "----------------------------------------------", "",
             "The accumulated Phase79–83 sequence tested probe optimization, optimizer steps, pair supervision, nonlinear readability and first-moment history.",
             "All assessment used training-role identities. No fresh validation, primary evaluation or model promotion occurred.",
             "Phase81 native and pair-supervision arms both retain raw and accepted main exact membership 0 of 1024. Pair supervision increases proposal background assignments.",
             "Phase83 passes 0 of 12 structural improvement gates; all conditions remain raw and accepted 0 of 64 on the 96-event audit.",
             "Zero-success intervals do not establish equivalence. Tiny versus main exposure remains confounded; undertraining is not proven.",
             "Physical trees, p4 closure and physical beam endpoints remain unavailable. Truth-assisted probes and conditional oracles are diagnostic, not deployable outputs.",
             "See :doc:`../../previous_studies` for the complete review, uncertainty, resource costs and preserved corrections.", "",
             ":download:`Integrity <previous-studies-integrity.json>`, :download:`inventory <previous-studies-inventory.json>` and :download:`decoder <previous-studies-decoder.txt>`.", ""]
    for record in value["metric_downloads"]:
        name = record["filename"]
        lines += [f":download:`{name} <{name}>`.", ""]
    return lines
