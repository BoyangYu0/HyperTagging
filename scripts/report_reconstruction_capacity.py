#!/usr/bin/env python
"""Generate an every-level, target-policy-specific production capacity report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypertagging.data.capacity import production_capacity_report
from hypertagging.training.model_config import MODEL_PRESETS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), default="production_baseline")
    parser.add_argument("--target-policy", choices=("complete_only", "reconstructable_partial"), default="complete_only")
    parser.add_argument("--output", type=Path, default=Path("/tmp/hypertagging-capacity-report.json"))
    args = parser.parse_args()
    index = json.loads(args.dataset_index.read_text(encoding="utf-8"))
    architecture = MODEL_PRESETS[args.model_preset]
    report = production_capacity_report(
        index,
        global_n_queries=architecture.n_queries,
        global_max_cardinality=architecture.max_cardinality,
        n_queries_by_level=dict(architecture.n_queries_by_level),
        max_cardinality_by_level=dict(architecture.max_cardinality_by_level),
        target_policy=args.target_policy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    if not report["production_training_allowed"]:
        raise OverflowError("target-policy-specific production capacity overflow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
