# 10M RI mDST production evidence

This directory is the compact, reviewable Git bundle for the completed 10M
run-independent mDST production. The dataset itself remains outside Git at:

```text
/data/dust/user/boyangyu/hypertagging/production_10m_ri_all_exp_20260811_f4e54df
```

The scanned `MC16ri_run2` release contains one available experiment (`e1004`)
and seven physics categories. Production completed with 10,000,000 validated
events, 10,000,000 globally unique event IDs, 2,000 complete shards, and no
failed publications.

## Review entry points

- [`validate_mdst_10m_campaign.executed.ipynb`](validate_mdst_10m_campaign.executed.ipynb)
  is the executed numerical and visual validation notebook.
- [`inspect_production_manifest.executed.ipynb`](inspect_production_manifest.executed.ipynb)
  is the executed manifest and publication-contract audit.
- [`reports/validation_report.html`](reports/validation_report.html) is the
  standalone validation report.
- [`reports/DATASET_CARD.md`](reports/DATASET_CARD.md) summarizes the immutable
  dataset contract and downstream entry points.
- [`reports/final_validation.json`](reports/final_validation.json) contains the
  authoritative validation numbers.
- [`reports/campaign_metadata.json`](reports/campaign_metadata.json) records
  source, Condor, schema, feature-contract, and artifact provenance.

The unexecuted notebook files are retained beside the executed copies so the
analysis can be reviewed or rerun. The `figures/` directory contains the
publication plots and representative/extreme event-tree visualizations. The
`reports/` directory includes the source-backed report artifact, SQLite source,
queries, delivery receipt, shard metrics, readiness record, and validation
method.

## Re-execution

The notebooks require access to the production root above and the frozen
source snapshot stored under its `source/f4e54df23b5c` directory. On the DESY
host they can be executed with the production Python environment and an empty
Jupyter configuration directory, as documented in the notebook setup cells.

Large payloads are intentionally excluded from Git: the 101.599 GiB shard set,
the 3.7 MiB task manifest, and the 6.5 MiB training dataset index remain with
the production dataset. Their hashes and summary metadata are captured in the
reports included here.
