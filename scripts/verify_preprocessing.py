#!/usr/bin/env python
"""Validate and inspect direct-mDST HyperTagging preprocessing output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import awkward as ak
import pyarrow.parquet as pq

from hypertagging.preprocessing.schema_v4 import iter_event_records_v4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Processed parquet file.")
    parser.add_argument("--event", type=int, default=0, help="Event index inside the processed file.")
    parser.add_argument(
        "--all-events",
        action="store_true",
        help="Run structural and four-vector checks on every event instead of only --event.",
    )
    parser.add_argument("--dump-tree", action="store_true", help="Print nodes grouped by level.")
    parser.add_argument("--check-p4", action="store_true", help="Check mother p4 equals daughter p4 sum.")
    parser.add_argument("--check-charge", action="store_true", help="Check reconstructed mother charge equals daughter charge sum.")
    parser.add_argument("--check-tree", action="store_true", help="Check links, DAG, levels, and copy references.")
    parser.add_argument("--check-pid", action="store_true", help="Print PID distributions before/after if present.")
    parser.add_argument("--all", action="store_true", help="Run all checks.")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="Absolute/relative p4 tolerance.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    parquet = pq.ParquetFile(args.input)
    if "event_json" in parquet.schema_arrow.names:
        # Native v4 stores one event per row.  Keep the CLI bounded by the
        # producer row group instead of materializing an entire shard.
        events = iter_event_records_v4(args.input)
        event_count = parquet.metadata.num_rows
        sidecar = Path(args.input).with_suffix(Path(args.input).suffix + ".metadata.json")
        metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        payload = {"summary_json": json.dumps({"pid_summary": metadata.get(
            "preprocessing_configuration", {}
        ).get("pid_filter", metadata.get("aggregate_pid_statistics", {}))})}
    else:
        payload = ak.to_list(ak.from_parquet(args.input))[0]
        events = iter(payload["events"])
        event_count = len(payload["events"])
    if args.event < 0 or args.event >= event_count:
        raise IndexError(f"event index {args.event} out of range for {event_count} events")

    run_all = args.all or not any([
        args.dump_tree, args.check_p4, args.check_charge, args.check_tree, args.check_pid
    ])
    checked_count = 0
    for index, event in enumerate(events):
        if index == args.event and (args.dump_tree or run_all):
            dump_tree(event)
        if not args.all_events and index != args.event:
            continue
        if args.check_tree or run_all:
            check_tree(event, verbose=not args.all_events)
        if args.check_p4 or run_all:
            check_p4(event, tolerance=args.tolerance, verbose=not args.all_events)
        if args.check_charge or run_all:
            check_charge(event, tolerance=args.tolerance, verbose=not args.all_events)
        checked_count += 1
        if not args.all_events:
            break
    if args.check_pid or run_all:
        print_pid_summary(payload)
    if args.all_events and (args.check_tree or args.check_p4 or args.check_charge or run_all):
        print(f"validated events={checked_count}")
    return 0


def dump_tree(event: dict[str, object]) -> None:
    nodes = _nodes_by_id(event)
    print(f"event_id={event['event_id']} roots={event['root_ids']}")
    for level_record in event["levels"]:
        print(f"level {level_record['level']}:")
        for node_id in level_record["node_ids"]:
            node = nodes[node_id]
            print(
                "  "
                f"id={node_id} pdg={node['pdg']} token={node['token']} "
                f"p4=({node['px']:.6g},{node['py']:.6g},{node['pz']:.6g},{node['energy']:.6g}) "
                f"m={node['mass']:.6g} parent={node['parent_id']} daughters={node['daughter_ids']} "
                f"copied_from={node['copied_from']} flags={node['flags']}"
            )


def check_tree(event: dict[str, object], *, verbose: bool = True) -> None:
    nodes = _nodes_by_id(event)
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(node_id: int) -> None:
        if node_id in visiting:
            raise ValueError(f"cycle detected at node {node_id}")
        if node_id in visited:
            return
        visiting.add(node_id)
        node = nodes[node_id]
        for child_id in node["daughter_ids"]:
            if child_id not in nodes:
                raise ValueError(f"missing daughter {child_id} from node {node_id}")
            child = nodes[child_id]
            if child["parent_id"] != node_id:
                raise ValueError(f"parent mismatch for child {child_id}: {child['parent_id']} != {node_id}")
            if node["level"] <= child["level"]:
                raise ValueError(f"level violation parent {node_id} child {child_id}")
            visit(child_id)
        parent_id = node["parent_id"]
        if parent_id != -1 and parent_id not in nodes:
            raise ValueError(f"missing parent {parent_id} from node {node_id}")
        copied_from = node["copied_from"]
        if copied_from != -1 and copied_from not in nodes:
            raise ValueError(f"missing copied_from {copied_from} from node {node_id}")
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in nodes:
        visit(node_id)
    if verbose:
        print(f"tree ok: nodes={len(nodes)}")


def check_p4(event: dict[str, object], *, tolerance: float, verbose: bool = True) -> None:
    nodes = _nodes_by_id(event)
    max_abs = 0.0
    max_rel = 0.0
    for node_id, node in nodes.items():
        daughters = node["daughter_ids"]
        if not daughters:
            continue
        summed = [
            sum(float(nodes[child_id][field]) for child_id in daughters)
            for field in ("px", "py", "pz", "energy")
        ]
        stored = [float(node[field]) for field in ("px", "py", "pz", "energy")]
        for actual, expected in zip(stored, summed):
            diff = abs(actual - expected)
            rel = diff / max(abs(expected), tolerance)
            max_abs = max(max_abs, diff)
            max_rel = max(max_rel, rel)
            if diff > tolerance and rel > tolerance:
                raise ValueError(f"p4 mismatch at node {node_id}: stored={stored} summed={summed}")
        mc_values = [node.get(field) for field in ("mc_px", "mc_py", "mc_pz", "mc_energy")]
        if all(value is not None for value in mc_values):
            if all(math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance) for a, b in zip(stored, mc_values)):
                print(f"warning: node {node_id} reco p4 numerically equals diagnostic MC p4")
    if verbose:
        print(f"p4 ok: max_abs={max_abs:.3g} max_rel={max_rel:.3g}")


def print_pid_summary(payload: dict[str, object]) -> None:
    summary_json = payload.get("summary_json") or "{}"
    summary = json.loads(summary_json)
    pid_summary = summary.get("pid_summary", {})
    print("pid summary:")
    print(json.dumps(pid_summary, indent=2, sort_keys=True))


def check_charge(event: dict[str, object], *, tolerance: float, verbose: bool = True) -> None:
    nodes = _nodes_by_id(event)
    for node_id, node in nodes.items():
        if not node["daughter_ids"]:
            continue
        summed = sum(float(nodes[child_id]["charge"]) for child_id in node["daughter_ids"])
        stored = float(node["charge"])
        if not math.isclose(stored, summed, rel_tol=tolerance, abs_tol=tolerance):
            raise ValueError(f"charge mismatch at node {node_id}: stored={stored} summed={summed}")
    if verbose:
        print("charge ok: recursive daughter sums")


def _nodes_by_id(event: dict[str, object]) -> dict[int, dict[str, object]]:
    return {int(node["node_id"]): node for node in event["nodes"]}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
