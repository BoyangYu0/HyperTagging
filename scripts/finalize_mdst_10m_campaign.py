#!/usr/bin/env python
"""Write the final dataset card, campaign metadata, and artifact checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment(path: str) -> str:
    matches = [part for part in Path(path).parts if re.fullmatch(r"e\d+", part)]
    if len(matches) != 1:
        raise ValueError(f"Expected one experiment component in {path!r}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.production_root.resolve()
    manifest_path = root / "manifests" / "mdst_10m_ri_all_exp.jsonl"
    validation_path = root / "validation" / "final_validation.json"
    if not manifest_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Manifest and exhaustive final validation are required")

    tasks = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    campaign_id = validation["campaign_id"]
    shard_dir = root / campaign_id / "shards"
    suffix_counts = {
        "parquet": len(list(shard_dir.glob("*.parquet"))),
        "metadata_json": len(list(shard_dir.glob("*.parquet.metadata.json"))),
        "result_json": len(list(shard_dir.glob("*.parquet.result.json"))),
        "completion_marker": len(list(shard_dir.glob("*.parquet.complete"))),
        "failure_json": len(list(shard_dir.glob("*.failure.json"))),
    }
    expected = validation["tasks"]
    if any(suffix_counts[key] != expected for key in (
        "parquet", "metadata_json", "result_json", "completion_marker"
    )) or suffix_counts["failure_json"]:
        raise RuntimeError(f"Publication companion mismatch: {suffix_counts}")

    relative_artifacts = [
        "manifests/mdst_10m_ri_all_exp.jsonl",
        "manifests/mdst_10m_ri_all_exp.summary.json",
        "manifests/production_readiness_10k_manual_signoff_20260811.json",
        "manifests/mdst_10m_preflight_task0.sub",
        "manifests/mdst_10m_bulk_missing_after_preflight.sub",
        "validation/final_validation.json",
        "validation/validation_method.json",
        "validation/dataset_index.json",
        "validation/condor_monitor_summary.json",
        "validation/notebook/validate_mdst_10m_campaign.ipynb",
        "validation/notebook/validate_mdst_10m_campaign.executed.ipynb",
        "validation/notebook/inspect_production_manifest.ipynb",
        "validation/notebook/inspect_production_manifest.executed.ipynb",
        "validation/notebook/manifest_figures/manifest_categories.png",
        "validation/notebook/manifest_figures/production_manifest_report.json",
        "validation/notebook/figures/shard_metrics.csv",
        "validation/notebook/figures/notebook_takeaways.json",
        "validation/notebook/figures/category_coverage.png",
        "validation/notebook/figures/throughput_memory.png",
        "validation/notebook/figures/topology_leaf_modes.png",
        "validation/notebook/figures/klm_by_category.png",
        "validation/notebook/figures/representative_category_trees.png",
        "validation/notebook/figures/extreme_topology_trees.png",
        "validation/report/artifact.json",
        "validation/report/report_data.sqlite",
        "validation/report/report_queries.sql",
        "validation/report/validation_report.html",
        "validation/report/delivery_receipt.json",
    ]
    artifacts: dict[str, dict[str, Any]] = {}
    for relative in relative_artifacts:
        path = root / relative
        if path.is_file():
            artifacts[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    missing_artifacts = sorted(set(relative_artifacts) - set(artifacts))
    if missing_artifacts:
        raise FileNotFoundError(
            "Finalization artifacts are missing: " + ", ".join(missing_artifacts)
        )

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    metadata = {
        "generated_at": generated_at,
        "dataset_root": str(root),
        "campaign_id": campaign_id,
        "campaign_config_digest": validation["campaign_config_digest"],
        "campaign_stage": validation["campaign_stage"],
        "planned_events": validation["planned_events"],
        "validated_events": validation["validated_events"],
        "unique_event_uids": validation["unique_event_uids"],
        "tasks": validation["tasks"],
        "publication_files": suffix_counts,
        "output_bytes": validation["output_bytes"],
        "output_bytes_per_event": validation["output_bytes_per_event"],
        "input_release": "MC16ri_run2",
        "ri_path_violations": sum(
            "MC16ri_run2" not in Path(task["input_file"]).parts for task in tasks
        ),
        "experiments": sorted({experiment(task["input_file"]) for task in tasks}),
        "physics_categories": validation["category_distribution"],
        "unique_input_files": len({task["input_file"] for task in tasks}),
        "source": {
            "git_commit": validation["source_git_commit"],
            "git_tree": validation["source_git_tree"],
            "state": validation["source_state"],
            "snapshot": f"source/{validation['source_git_commit'][:12]}",
        },
        "contracts": {
            "schema_version": validation["schema_version"],
            "pid_vocabulary_version": validation["pid_vocabulary_version"],
            "feature_spec_hash": validation["feature_spec_hash"],
            "model_feature_contract_hash": validation[
                "model_feature_contract_hash"
            ],
            "track_fit_policy": validation["track_fit_policy"],
            "klm_training_scope": validation["klm_training_scope"],
        },
        "readiness": {
            "manual_10k_middle_verification": "completed (user assertion)",
            "reviewer_identity": "not supplied",
            "report_sha256": validation["production_readiness_report_sha256"],
        },
        "condor": {
            "initial_preflight_cluster": {
                "id": 4844425,
                "accepted": False,
                "reason": "Unix environment separator syntax rejected before processing",
            },
            "successful_preflight_cluster": {"id": 4844426, "accepted": True},
            "bulk_cluster": {"id": 4844428, "accepted": True},
        },
        "validation": {
            "all_completion_markers_valid": validation[
                "all_completion_markers_valid"
            ],
            "global_uid_validation_passes": validation[
                "global_uid_validation_passes"
            ],
            "missing_shards": validation["missing_shards"],
            "incomplete_reconstructable_branches": validation[
                "incomplete_reconstructable_branches"
            ],
        },
        "artifacts": artifacts,
    }
    metadata_path = root / "campaign_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_path = root / "checksums.sha256"
    checksum_path.write_text(
        "".join(
            f"{record['sha256']}  {relative}\n"
            for relative, record in sorted(artifacts.items())
        ),
        encoding="utf-8",
    )
    card_path = root / "DATASET_CARD.md"
    category_lines = "\n".join(
        f"- `{category}`: {events:,} events"
        for category, events in sorted(validation["category_distribution"].items())
    )
    card_path.write_text(
        f"""# 10M run-independent mDST graph dataset

Status: **validated and ready**  
Campaign: `{campaign_id}`  
Generated: `{generated_at}`

## Dataset

- Events: {validation['validated_events']:,} validated / {validation['planned_events']:,} planned
- Unique event UIDs: {validation['unique_event_uids']:,}
- Shards: {validation['completed_shards']:,} Parquet files plus metadata, result, and completion sidecars
- Input: `MC16ri_run2` only; experiments `{', '.join(metadata['experiments'])}`
- Unique source files: {metadata['unique_input_files']:,}
- Output: {validation['output_bytes'] / 2**30:.3f} GiB ({validation['output_bytes_per_event']:.1f} bytes/event)

## Physics-category allocation

{category_lines}

## Immutable production contract

- Source commit: `{validation['source_git_commit']}` (clean tree `{validation['source_git_tree']}`)
- Schema: `{validation['schema_version']}`
- Feature spec: `{validation['feature_spec_hash']}`
- Model feature contract: `{validation['model_feature_contract_hash']}`
- KLM training scope: `{validation['klm_training_scope']}`

## Entry points

- Dataset manifest: `manifests/mdst_10m_ri_all_exp.jsonl`
- Exhaustive validation: `validation/final_validation.json`
- Portable report: `validation/report/validation_report.html`
- Executed campaign notebook: `validation/notebook/validate_mdst_10m_campaign.executed.ipynb`
- Executed manifest notebook: `validation/notebook/inspect_production_manifest.executed.ipynb`
- Machine-readable metadata: `campaign_metadata.json`
- Artifact checksums: `checksums.sha256`

Use the manifest as the dataset entry point. Preserve every per-shard sidecar when copying the dataset.
""",
        encoding="utf-8",
    )
    print(metadata_path)
    print(card_path)
    print(checksum_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
