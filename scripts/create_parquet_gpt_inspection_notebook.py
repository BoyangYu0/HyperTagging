#!/usr/bin/env python
"""Create the direct-mDST parquet inspection and GPT-like smoke-test notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import nbformat as nbf


DEFAULT_INPUT = Path("/data/dust/user/boyangyu/hypertagging/preprocess_mc16ri_run2_100.parquet")
DEFAULT_MANIFEST_SUMMARY = Path(
    "/data/dust/user/boyangyu/hypertagging/production_10m/manifests/mdst_10m.summary.json"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "inspect_preprocessed_parquet_and_gpt_like.ipynb"
)


def markdown(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def build_notebook(default_input: Path, manifest_summary: Path) -> nbf.NotebookNode:
    cells = [
        markdown(
            """
            # Inspect direct-mDST parquet and test the GPT-like model

            ## Goal

            This historical-compatibility notebook is an executable tour of the
            retired `direct-mdst-tree-v1` parquet contract. It does not describe
            the production schema-v4 path. It:

            1. demonstrates the top-level, event, level, node, legacy, and summary records;
            2. checks tree links, provenance IDs, numerical fields, and four-vector closure;
            3. converts variable-width direct trees into a teacher-forced GPT-like batch;
            4. runs `ParticleEmbedder` and `MultiGPT` on real preprocessed events; and
            5. executes one CPU optimizer step to verify forward, loss, gradient, and update paths.

            The neural networks are randomly initialized. Losses and predictions below are
            software-integration diagnostics, **not physics-performance measurements**.
            """
        ),
        markdown(
            """
            ## Setup

            Run with the project CPU environment:

            ```bash
            source /data/dust/user/boyangyu/uv_env/bin/activate
            ```

            Override the default parquet with `HYPERTAGGING_PREPROCESS_OUTPUT`.
            The notebook uses four events for the model smoke test, so it remains safe on a
            CPU login node.
            """
        ),
        code(
            f"""
            from pathlib import Path
            import json
            import os

            import awkward as ak
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import torch

            from hypertagging.data.notebook_fixtures import write_notebook_fixture_v1
            from hypertagging.data.direct_gpt import (
                DIRECT_FEATURE_NAMES,
                build_direct_multi_gpt_batch,
                collate_direct_gpt_events,
                load_direct_gpt_events,
            )
            from hypertagging.losses.gpt_losses import distance, radius_loss
            from hypertagging.losses.link_losses import link_metrics
            from hypertagging.models.gpt_like import MultiGPT, ParticleEmbedder

            configured_path = (
                os.environ.get("HYPERTAGGING_PARQUET")
                or os.environ.get("HYPERTAGGING_PREPROCESS_OUTPUT")
            )
            if configured_path:
                INPUT_PATH = Path(configured_path)
                if not INPUT_PATH.exists():
                    raise FileNotFoundError(f"Configured parquet does not exist: {{INPUT_PATH}}")
                FIXTURE_MODE = False
            else:
                INPUT_PATH = Path("/tmp/hypertagging-direct-gpt-fixture.parquet")
                write_notebook_fixture_v1(INPUT_PATH)
                FIXTURE_MODE = True
            MANIFEST_SUMMARY_PATH = Path({str(manifest_summary)!r})
            FIGURE_DIR = Path(os.environ.get("HYPERTAGGING_FIGURE_DIR", "/tmp/hypertagging-direct-gpt-figures"))
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            MODEL_EVENT_LIMIT = 4
            DEVICE = torch.device("cpu")
            SEED = int(os.environ.get("HYPERTAGGING_NOTEBOOK_SEED", "20260730"))
            torch.manual_seed(SEED)
            np.random.seed(SEED)

            print(f"Input: {{INPUT_PATH}}")
            print("Mode:", "SOFTWARE FIXTURE — NOT PHYSICS PERFORMANCE" if FIXTURE_MODE else "REAL PREPROCESSED DATA")
            print(f"PyTorch: {{torch.__version__}}; device: {{DEVICE}}; CUDA available: {{torch.cuda.is_available()}}")
            """
        ),
        markdown("## Steps"),
        markdown("### 1. Load the parquet and inspect its top-level structure"),
        code(
            """
            awkward_payload = ak.from_parquet(INPUT_PATH)
            payload = ak.to_list(awkward_payload)[0]
            required = {"schema_version", "events", "legacy_levels", "summary_json"}
            missing = sorted(required - set(payload))
            if missing:
                raise KeyError(f"Missing required parquet fields: {missing}")
            production_summary = json.loads(payload["summary_json"])

            print("Awkward type:")
            print(akward_type := ak.type(awkward_payload))
            print("\\nTop-level fields:", list(payload))
            print("Schema version:", payload["schema_version"])
            print("Events:", len(payload["events"]))
            print("Legacy level rows:", len(payload["legacy_levels"]))
            """
        ),
        markdown("### 2. Inspect event, level, and node records"),
        code(
            """
            event = payload["events"][0]
            event_overview = {
                key: event[key]
                for key in (
                    "event_id",
                    "event_uid",
                    "experiment",
                    "run",
                    "production",
                    "root_ids",
                )
            }
            display(pd.Series(event_overview, name="first event"))

            display(pd.DataFrame(event["levels"]).rename(columns={"node_ids": "node_ids at level"}))

            node_columns = [
                "node_id",
                "pdg",
                "token",
                "level",
                "parent_id",
                "daughter_ids",
                "reco_id",
                "energy",
                "px",
                "py",
                "pz",
                "mass",
                "mc_energy",
                "flags",
            ]
            display(pd.DataFrame(event["nodes"])[node_columns].head(12))
            """
        ),
        markdown("### 3. Inspect the legacy compatibility view and production summary"),
        code(
            """
            if payload["legacy_levels"]:
                legacy_row = payload["legacy_levels"][0]
                print("Legacy row fields:", list(legacy_row))
                display(
                    pd.Series(
                        {
                            "evtNum": legacy_row["evtNum"],
                            "depth": legacy_row["depth"],
                            "seq_len": legacy_row["seq_len"],
                            "feature shape": np.asarray(legacy_row["feature"]).shape,
                            "motherIndex shape": np.asarray(legacy_row["motherIndex"]).shape,
                            "E_Rec": legacy_row["E_Rec"],
                        },
                        name="first legacy level row",
                    )
                )

            display(
                pd.Series(
                    {
                        "events": production_summary["events"],
                        "nodes before pruning": production_summary["nodes_before_pruning"],
                        "nodes after pruning": production_summary["nodes_after_pruning"],
                        "unmatched reco": production_summary["unmatched_reco"],
                        "track records": production_summary["collection"].get("track_records", 0),
                        "neutral ECL records": production_summary["collection"].get("ecl_records", 0),
                        "entry sequences": production_summary.get("entry_sequences"),
                    },
                    name="production summary",
                )
            )
            """
        ),
        markdown("### 4. Summarize dataset elements and validate the tree representation"),
        code(
            """
            event_rows = []
            node_rows = []
            max_p4_closure_error = 0.0
            for event_index, current_event in enumerate(payload["events"]):
                nodes_by_id = {int(node["node_id"]): node for node in current_event["nodes"]}
                event_rows.append(
                    {
                        "event_index": event_index,
                        "event_uid": current_event["event_uid"],
                        "nodes": len(current_event["nodes"]),
                        "levels": len(current_event["levels"]),
                        "roots": len(current_event["root_ids"]),
                    }
                )
                for node in current_event["nodes"]:
                    reco_source = node["reco_id"].split(":", 1)[0] if node["reco_id"] else "truth topology"
                    node_rows.append(
                        {
                            "event_index": event_index,
                            "level": int(node["level"]),
                            "pdg": int(node["pdg"]),
                            "reco_source": reco_source,
                            "is_leaf": not node["daughter_ids"],
                            "has_mc_p4": node["mc_energy"] is not None,
                            "flags": ",".join(node["flags"]) if node["flags"] else "none",
                        }
                    )
                    if node["daughter_ids"]:
                        for field in ("energy", "px", "py", "pz"):
                            daughter_sum = sum(
                                float(nodes_by_id[child_id][field])
                                for child_id in node["daughter_ids"]
                            )
                            max_p4_closure_error = max(
                                max_p4_closure_error,
                                abs(float(node[field]) - daughter_sum),
                            )

            events_frame = pd.DataFrame(event_rows)
            nodes_frame = pd.DataFrame(node_rows)
            dataset_summary = pd.Series(
                {
                    "events": len(events_frame),
                    "nodes": len(nodes_frame),
                    "minimum nodes/event": events_frame["nodes"].min(),
                    "maximum nodes/event": events_frame["nodes"].max(),
                    "mean nodes/event": events_frame["nodes"].mean(),
                    "maximum levels/event": events_frame["levels"].max(),
                    "unique event_uid": events_frame["event_uid"].nunique(),
                    "maximum parent p4 closure error": max_p4_closure_error,
                },
                name="dataset structure",
            )
            display(dataset_summary)
            display(pd.crosstab(nodes_frame["level"], nodes_frame["reco_source"], margins=True))

            assert events_frame["event_uid"].is_unique
            assert max_p4_closure_error < 1e-8
            """
        ),
        markdown("### 5. Convert real events to the direct-tree GPT batch"),
        code(
            """
            direct_events = load_direct_gpt_events(INPUT_PATH, limit=MODEL_EVENT_LIMIT)
            structure = collate_direct_gpt_events(direct_events)

            structure_table = pd.DataFrame(
                [
                    {
                        "field": name,
                        "shape": tuple(value.shape),
                        "dtype": str(value.dtype),
                        "finite": bool(torch.isfinite(value).all()) if value.dtype is not torch.bool else True,
                    }
                    for name, value in structure.items()
                ]
            )
            display(structure_table)
            print("Feature order:", DIRECT_FEATURE_NAMES)
            print(
                "Active nodes per event:",
                structure["padding_mask"].sum(dim=1).tolist(),
            )
            """
        ),
        markdown(
            """
            The historical GPT collator assumes a fixed level width. The direct-tree adapter
            instead uses reconstructed leaves as visible inputs and zero-valued slots for
            higher-level queries. Leaf queries can attend to leaves; each higher-level query
            can attend only to strictly lower levels. Targets remain the embeddings of the
            corresponding truth-guided nodes, and link labels are parent positions.
            """
        ),
        markdown("### 6. Embed particles and inspect the autoregressive attention mask"),
        code(
            """
            particle_embedder = ParticleEmbedder(
                n_features=len(DIRECT_FEATURE_NAMES),
                tr_width=16,
                tr_n_head=2,
                tr_n=1,
                tr_hidden_size=32,
                pdg_emb=4,
                dim_hyper=4,
                num_pdg=40,
                device=DEVICE,
            )
            particle_embeddings = particle_embedder(structure).detach()
            gpt_batch = build_direct_multi_gpt_batch(particle_embeddings, structure)

            first_count = int(structure["padding_mask"][0].sum())
            visible_count = int((structure["level_ids"][0, :first_count] == 0).sum())
            print("Particle embedding shape:", tuple(particle_embeddings.shape))
            print("First event active nodes:", first_count)
            print("First event visible leaf inputs:", visible_count)
            print("First event masked higher-level query inputs:", first_count - visible_count)

            shown = min(first_count, 50)
            allowed_attention = (~gpt_batch["src_mask"][0, :shown, :shown]).numpy()
            fig, axis = plt.subplots(figsize=(7, 6))
            image = axis.imshow(allowed_attention, cmap="Blues", interpolation="nearest", aspect="auto")
            axis.set_title("Allowed GPT attention for the first event")
            axis.set_xlabel("Key node position")
            axis.set_ylabel("Query node position")
            fig.colorbar(image, ax=axis, label="Allowed (1) / blocked (0)")
            fig.tight_layout()
            fig.savefig(FIGURE_DIR / "direct_gpt_attention_mask.png")
            plt.show()
            """
        ),
        markdown("### 7. Run MultiGPT, calculate losses, and execute one optimizer step"),
        code(
            """
            model = MultiGPT(
                rec_width=16,
                rec_n_head=2,
                rec_n=1,
                rec_hidden_size=32,
                link_width=16,
                link_n_head=2,
                link_n=1,
                link_hidden_size=32,
                dim_hyper=4,
                device=DEVICE,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            optimizer.zero_grad(set_to_none=True)

            predicted_embeddings, predicted_links = model(gpt_batch)
            active_mask = gpt_batch["lvl_code"].bool()
            link_mask = gpt_batch["links"] >= 0
            reconstruction_loss = distance(
                predicted_embeddings,
                gpt_batch["target"],
                active_mask,
            )
            link_loss, link_accuracy = link_metrics(
                predicted_links,
                gpt_batch["links"],
                link_mask,
            )
            radial_loss = radius_loss(
                predicted_embeddings,
                gpt_batch["mass"],
                link_mask,
            )
            total_loss = reconstruction_loss + link_loss + radial_loss
            total_loss.backward()

            finite_gradients = [
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            gradient_norm = torch.sqrt(
                sum(
                    torch.sum(parameter.grad.detach() ** 2)
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            )
            optimizer.step()

            display(
                pd.Series(
                    {
                        "predicted embedding shape": tuple(predicted_embeddings.shape),
                        "predicted link shape": tuple(predicted_links.shape),
                        "reconstruction loss": float(reconstruction_loss.detach()),
                        "link loss": float(link_loss.detach()),
                        "link accuracy (untrained)": float(link_accuracy.detach()),
                        "radial loss": float(radial_loss.detach()),
                        "total loss": float(total_loss.detach()),
                        "gradient L2 norm": float(gradient_norm),
                        "all gradients finite": all(finite_gradients),
                    },
                    name="real-parquet MultiGPT smoke test",
                )
            )
            """
        ),
        markdown("## Checks"),
        code(
            """
            checks = {
                "schema is direct-mdst-tree-v1": payload["schema_version"] == "direct-mdst-tree-v1",
                "event provenance is unique": events_frame["event_uid"].is_unique,
                "tree p4 closure passes": max_p4_closure_error < 1e-8,
                "GPT output embeddings finite": bool(torch.isfinite(predicted_embeddings).all()),
                "GPT link scores finite": bool(torch.isfinite(predicted_links).all()),
                "loss finite": bool(torch.isfinite(total_loss)),
                "gradients finite": all(finite_gradients),
                "optimizer step completed": True,
            }
            display(pd.Series(checks, name="validation checks"))
            assert all(checks.values())
            """
        ),
        markdown("## Next Steps"),
        code(
            """
            if MANIFEST_SUMMARY_PATH.exists():
                manifest_summary = json.loads(MANIFEST_SUMMARY_PATH.read_text())
                display(pd.Series(manifest_summary, name="planned large-scale production"))
            else:
                print("10M-event manifest has not been generated in this environment.")
            """
        ),
        markdown(
            """
            - The production plan targets **10 million input events**, which is deliberately
              more conservative than counting level rows as training samples.
            - The current adapter is suitable for integration testing and teacher-forced
              experimentation. Before a full training campaign, establish feature
              normalization, train/validation/test splits by `event_uid` and source file,
              checkpointing, and physics metrics.
            - The newer set-based level-autoregressive model remains the preferred architecture
              when permutation invariance and variable output cardinality are primary goals.
            """
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3 (HyperTagging CPU)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    return notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest-summary", type=Path, default=DEFAULT_MANIFEST_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(
        build_notebook(args.input.resolve(), args.manifest_summary.resolve()),
        args.output,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
