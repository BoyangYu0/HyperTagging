"""Lossless, authenticated publication of the frozen Phase79–83 evidence."""
import base64
import hashlib
import lzma
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


def expand_sources(roots, inventory, external_sources=None):
    policy = inventory["deduplication"]
    if policy["format"] != "exact-shared-subtrees-v1" or policy["reference_key"] != "$ref":
        raise ValueError("Unknown previous-studies shared-subtree format")
    definitions = roots.pop(policy["definitions_source"])
    if not isinstance(definitions, list) or len(definitions) != policy["definitions"]:
        raise ValueError("Previous-studies definition cardinality mismatch")
    external_definitions = roots.pop("external-subtree-definitions", [])
    if len(external_definitions) != inventory.get("external_reference_count", 0):
        raise ValueError("Previous-studies external reference count mismatch")
    external_sources = external_sources or {}
    active, cache, external_cache = set(), {}, {}
    def external(index):
        if type(index) is not int or not 0 <= index < len(external_definitions):
            raise ValueError("Invalid previous-studies external reference")
        if index not in external_cache:
            record = external_definitions[index]
            if record["file"] not in external_sources or not isinstance(record["path"], list) or len(record["path"]) > 128:
                raise ValueError("Invalid previous-studies external path")
            value = external_sources[record["file"]]
            for key in record["path"]:
                if isinstance(value, list):
                    if type(key) is not int or not 0 <= key < len(value):
                        raise ValueError("Invalid previous-studies external array path")
                elif not isinstance(value, dict) or not isinstance(key, str) or key not in value:
                    raise ValueError("Invalid previous-studies external object path")
                value = value[key]
            data = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("Previous-studies external subtree hash mismatch")
            external_cache[index] = value
        return external_cache[index]
    def expand(value):
        if isinstance(value, dict):
            if set(value) == {"$external"}: return external(value["$external"])
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


def load_external_sources(output, records):
    expected = {"phase78-review.json", *(f"phase78-events-{i}.json" for i in range(3))}
    if len(records) != 4 or {r["filename"] for r in records} != expected:
        raise ValueError("Previous-studies external source count or names changed")
    codec = sibling("wiki_phase72_compact")
    privacy = sibling("wiki_privacy")
    values = {}
    for record in records:
        path = output / record["filename"]
        if path.is_symlink() or path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Unsafe previous-studies external source")
        raw = path.read_bytes()
        if digest(raw) != record["envelope_sha256"]:
            raise ValueError("Previous-studies external envelope hash mismatch")
        decoded = codec.decode(json.loads(raw), LIMIT)
        if len(decoded) != record["decoded_bytes"] or digest(decoded) != record["decoded_sha256"]:
            raise ValueError("Previous-studies external decoded hash mismatch")
        value = privacy._strict_json(decoded.decode())
        if privacy._contains_private_fields(value) or privacy.redact(decoded.decode()) != decoded.decode():
            raise ValueError("Private previous-studies external source")
        values[record["filename"]] = value
    return values


def decode_fragments(wrapper):
    if set(wrapper) != {"encoding", "decoded_bytes", "decoded_sha256", "data"} or wrapper["encoding"] != "bounded-xz-base32-fragments-v1":
        raise ValueError("Unknown previous-studies inner encoding")
    size = wrapper["decoded_bytes"]
    if type(size) is not int or not 0 <= size <= 5_000_000 or not isinstance(wrapper["data"], str) or len(wrapper["data"]) > 10 * 1024 * 1024:
        raise ValueError("Previous-studies inner byte limit")
    try:
        packed = base64.b32decode(wrapper["data"], casefold=False)
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=64 * 1024 * 1024)
        decoded = decoder.decompress(packed, max_length=size + 1)
    except (lzma.LZMAError, ValueError) as error:
        raise ValueError("Previous-studies inner stream or memory limit") from error
    if len(decoded) != size or not decoder.eof or decoder.unused_data or decoder.check != lzma.CHECK_CRC64:
        raise ValueError("Previous-studies inner expansion or stream mismatch")
    if hashlib.sha256(decoded).hexdigest() != wrapper["decoded_sha256"]:
        raise ValueError("Previous-studies inner decoded hash mismatch")
    return decoded


def generate(root, output):
    source = Path(root) / SOURCE
    if not source.exists():
        return None
    binding_path = source / "binding.json"
    if binding_path.is_symlink() or binding_path.stat().st_size > LIMIT:
        raise ValueError("Unsafe previous-studies binding")
    binding = json.loads(binding_path.read_text())
    if (binding["version"] != "previous-studies-public-binding-v1"
            or binding["inner_encoding"] != "bounded-xz-base32-fragments-v1"
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
    if binding["inventory"]["file"] != "inventory.json.gz":
        raise ValueError("Unsafe previous-studies inventory name")
    inventory_path = source / "inventory.json.gz"
    if inventory_path.is_symlink() or inventory_path.stat().st_size > LIMIT:
        raise ValueError("Unsafe previous-studies inventory")
    inventory_packed = inventory_path.read_bytes()
    if digest(inventory_packed) != binding["inventory"]["compressed_sha256"]:
        raise ValueError("Previous-studies inventory compressed hash mismatch")
    inventory_decoder = zlib.decompressobj(31)
    inventory_bytes = inventory_decoder.decompress(inventory_packed, LIMIT + 1)
    if (len(inventory_bytes) > LIMIT or not inventory_decoder.eof
            or inventory_decoder.unused_data or inventory_decoder.unconsumed_tail
            or len(inventory_bytes) != binding["inventory"]["decoded_bytes"]):
        raise ValueError("Previous-studies inventory expansion or stream mismatch")
    if digest(inventory_bytes) != binding["inventory"]["sha256"]:
        raise ValueError("Previous-studies inventory hash mismatch")
    inventory = privacy._strict_json(inventory_bytes.decode())
    if privacy._contains_private_fields(inventory) or privacy.redact(inventory_bytes.decode()) != inventory_bytes.decode():
        raise ValueError("Private previous-studies inventory")
    if inventory["external_sources"] != binding["external_sources"]:
        raise ValueError("Previous-studies external source registry mismatch")
    external_values = load_external_sources(output, binding["external_sources"])
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
        fragment_data = decode_fragments(privacy._strict_json(data.decode()))
        if len(fragment_data) != part["fragment_decoded_bytes"] or digest(fragment_data) != part["fragment_decoded_sha256"]:
            raise ValueError("Previous-studies fragment binding mismatch")
        value = privacy._strict_json(fragment_data.decode())
        if privacy._contains_private_fields(value) or privacy.redact(fragment_data.decode()) != fragment_data.decode():
            raise ValueError("Private previous-studies decoded fragments")
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
                        "fragments": len(value["fragments"]), "fragment_decoded_bytes": len(fragment_data),
                        "fragment_decoded_sha256": digest(fragment_data)})
    reconstructed = expand_sources(reconstructed, inventory, external_values)
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
                 "inventory_sha256": digest(inventory_bytes), "external_sources": binding["external_sources"], "parts": records,
                 "fragment_count": len(fragment_ids), "all_supported_metrics_retained": True}
    (output / "previous-studies-integrity.json").write_bytes(canonical(integrity))
    (output / "previous-studies-decoder.txt").write_text(DECODER)
    return {"status": "COMPLETED", "stage": "training_only_sequential_evidence", "primary_eligible": False,
            "source_sha": binding["source_sha"], "public_summary": binding["public_summary"],
            "metric_downloads": records, "fragment_count": len(fragment_ids)}


DECODER = '''# Python 3: verify and decode every complete metric fragment envelope.
import base64, hashlib, json, pathlib, zlib, lzma
'''
DECODER += '\ndef decode_fragments(wrapper):\n    if set(wrapper) != {"encoding", "decoded_bytes", "decoded_sha256", "data"} or wrapper["encoding"] != "bounded-xz-base32-fragments-v1":\n        raise ValueError("Unknown previous-studies inner encoding")\n    size = wrapper["decoded_bytes"]\n    if type(size) is not int or not 0 <= size <= 5_000_000 or not isinstance(wrapper["data"], str) or len(wrapper["data"]) > 10 * 1024 * 1024:\n        raise ValueError("Previous-studies inner byte limit")\n    try:\n        packed = base64.b32decode(wrapper["data"], casefold=False)\n        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=64 * 1024 * 1024)\n        decoded = decoder.decompress(packed, max_length=size + 1)\n    except (lzma.LZMAError, ValueError) as error:\n        raise ValueError("Previous-studies inner stream or memory limit") from error\n    if len(decoded) != size or not decoder.eof or decoder.unused_data or decoder.check != lzma.CHECK_CRC64:\n        raise ValueError("Previous-studies inner expansion or stream mismatch")\n    if hashlib.sha256(decoded).hexdigest() != wrapper["decoded_sha256"]:\n        raise ValueError("Previous-studies inner decoded hash mismatch")\n    return decoded\n\n'
DECODER += '''
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
    value = decode_fragments(json.loads(value))
    assert len(value) == part["fragment_decoded_bytes"]
    assert hashlib.sha256(value).hexdigest() == part["fragment_decoded_sha256"]
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


def expand_sources(roots, inventory, external_sources=None):
    policy = inventory["deduplication"]
    if policy["format"] != "exact-shared-subtrees-v1" or policy["reference_key"] != "$ref":
        raise ValueError("Unknown previous-studies shared-subtree format")
    definitions = roots.pop(policy["definitions_source"])
    if not isinstance(definitions, list) or len(definitions) != policy["definitions"]:
        raise ValueError("Previous-studies definition cardinality mismatch")
    external_definitions = roots.pop("external-subtree-definitions", [])
    if len(external_definitions) != inventory.get("external_reference_count", 0):
        raise ValueError("Previous-studies external reference count mismatch")
    external_sources = external_sources or {}
    active, cache, external_cache = set(), {}, {}
    def external(index):
        if type(index) is not int or not 0 <= index < len(external_definitions):
            raise ValueError("Invalid previous-studies external reference")
        if index not in external_cache:
            record = external_definitions[index]
            if record["file"] not in external_sources or not isinstance(record["path"], list) or len(record["path"]) > 128:
                raise ValueError("Invalid previous-studies external path")
            value = external_sources[record["file"]]
            for key in record["path"]:
                if isinstance(value, list):
                    if type(key) is not int or not 0 <= key < len(value):
                        raise ValueError("Invalid previous-studies external array path")
                elif not isinstance(value, dict) or not isinstance(key, str) or key not in value:
                    raise ValueError("Invalid previous-studies external object path")
                value = value[key]
            data = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("Previous-studies external subtree hash mismatch")
            external_cache[index] = value
        return external_cache[index]
    def expand(value):
        if isinstance(value, dict):
            if set(value) == {"$external"}: return external(value["$external"])
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
assert inventory["external_sources"] == integrity["external_sources"]
assert len(integrity["external_sources"]) == 4
assert {r["filename"] for r in integrity["external_sources"]} == {"phase78-review.json", *("phase78-events-" + str(i) + ".json" for i in range(3))}
external_sources = {}
for record in integrity["external_sources"]:
    raw = pathlib.Path(record["filename"]).read_bytes()
    assert len(raw) <= 10485760 and hashlib.sha256(raw).hexdigest() == record["envelope_sha256"]
    e = json.loads(raw)
    assert set(e) == {"encoding", "decoded_bytes", "decoded_sha256", "data"}
    assert e["encoding"] == "bounded-gzip-base32-json-v1"
    assert type(e["decoded_bytes"]) is int and 0 <= e["decoded_bytes"] <= 5000000
    assert len(e["data"]) <= 10485760
    d = zlib.decompressobj(31)
    value = d.decompress(base64.b32decode(e["data"], casefold=False), e["decoded_bytes"] + 1)
    assert d.eof and not d.unused_data and not d.unconsumed_tail
    assert len(value) == e["decoded_bytes"] == record["decoded_bytes"]
    assert hashlib.sha256(value).hexdigest() == e["decoded_sha256"] == record["decoded_sha256"]
    external_sources[record["filename"]] = json.loads(value)
roots = {}
for part in integrity["parts"]:
    decoded = pathlib.Path(part["filename"].replace(".json", "-decoded.json"))
    for fragment in json.loads(decoded.read_bytes())["fragments"]:
        insert_fragment(roots, fragment)
roots = expand_sources(roots, inventory, external_sources)
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
    lines += ["Reconstruction also requires the unchanged Phase78 review and three event downloads, verified by the same standard-library decoder.", ""]
    for filename in ("phase78-review.json", *(f"phase78-events-{i}.json" for i in range(3))):
        lines += [f":download:`{filename} <{filename}>`.", ""]
    for record in value["metric_downloads"]:
        name = record["filename"]
        lines += [f":download:`{name} <{name}>`.", ""]
    return lines
