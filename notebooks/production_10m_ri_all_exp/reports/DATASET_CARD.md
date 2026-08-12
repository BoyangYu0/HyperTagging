# 10M run-independent mDST graph dataset

Status: **validated and ready**  
Campaign: `campaign-fb070c6c9805-f4e54df23b5c`  
Generated: `2026-08-11T11:24:44.630651Z`

## Dataset

- Events: 10,000,000 validated / 10,000,000 planned
- Unique event UIDs: 10,000,000
- Shards: 2,000 Parquet files plus metadata, result, and completion sidecars
- Input: `MC16ri_run2` only; experiments `e1004`
- Unique source files: 1,526
- Output: 101.599 GiB (10909.1 bytes/event)

## Physics-category allocation

- `ccbar`: 2,160,000 events
- `charged`: 1,315,000 events
- `ddbar`: 795,000 events
- `mixed`: 1,275,000 events
- `ssbar`: 770,000 events
- `taupair`: 1,570,000 events
- `uubar`: 2,115,000 events

## Immutable production contract

- Source commit: `f4e54df23b5c60115e475c5d68df4651899d678e` (clean tree `b6e3a4118b960e3a4676a61af9601438d56cef96`)
- Schema: `direct-mdst-tree-v4`
- Feature spec: `01dc6c9bdf1e9c3f2b7675c102cb26912a1ff904f1138658b4c3dcf69b57be12`
- Model feature contract: `546bc22b8f98b2924b402bcc79c479d997fb50026e88f9955a7ac99ce858e76e`
- KLM training scope: `included`

## Entry points

- Dataset manifest: `manifests/mdst_10m_ri_all_exp.jsonl`
- Exhaustive validation: `validation/final_validation.json`
- Portable report: `validation/report/validation_report.html`
- Executed campaign notebook: `validation/notebook/validate_mdst_10m_campaign.executed.ipynb`
- Executed manifest notebook: `validation/notebook/inspect_production_manifest.executed.ipynb`
- Machine-readable metadata: `campaign_metadata.json`
- Artifact checksums: `checksums.sha256`

Use the manifest as the dataset entry point. Preserve every per-shard sidecar when copying the dataset.
