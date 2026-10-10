# Repository storage and recovery

The 2026-09-28 cleanup preserves scientific evidence while removing expanded
dashboard allowlists and generated report payloads from the active Git tree.
`configs/repository_artifacts.json` records each original path, byte count,
SHA-256 and relative data-volume location, bound to source commit
`db1ff41467aba5080b04bd89ae532fbdbd8788c2`.

The archive contains 74 verified files (90,363,060 bytes). The 58 dashboard
registries remain portable as deterministic gzip files (2,978,596 bytes),
preserving every original byte after decompression. Sixteen generated report
outputs are available from the data archive. This removes 87,384,464 bytes
from the tracked payload before adding the small manifest, tools and review.

Raw study receipts, immutable selections, source snapshots, report source,
tests and historical audits stay in Git. They are used by provenance checks,
historical reproductions and the offline dashboard; file age alone is not a
reason to delete them. This cleanup does not rewrite Git history or reclaim
historical blobs from the remote object database. Old checkouts and active
training sources are not removed.

The original source commit also remains a recovery source in Git history if
the external archive is unavailable. Its full revision and each original
repository path are recorded in the manifest.

## Data location

On the current institute host, set:

```bash
export HYPERTAGGING_DATA_ROOT=/project/agkuhr/users/boyang/data/HyperTagging_artifacts
```

Other hosts can copy the `repository_archive/` subtree to their own data
volume and set the variable accordingly. The manifest stores relative paths;
neither credentials nor an automatic remote download is provided. Existing
production datasets/checkpoints remain at their authenticated original paths.

## Verify and restore

A fresh clone builds the dashboard without the archive or a data mount:

```bash
python scripts/manage_repository_artifacts.py verify
```

With `HYPERTAGGING_DATA_ROOT` set, the same command verifies all external
originals, including the report outputs. To recover a complete report package,
restore its archived outputs into the checkout beside its retained builders
and metadata (restored paths are ignored by Git):

```bash
python scripts/manage_repository_artifacts.py restore --output-dir .
```

Restoration validates hashes first and refuses to overwrite different existing
bytes. Use `--path` repeatedly to select exact manifest paths, or choose an
external `--output-dir` to obtain the original directory structure there.
Without a data root, only the portable registries can be restored.

Historical registry generators still produce their original `.json` files.
Identical restored copies are accepted. If a generator changes a registry,
the dashboard rejects disagreement with the packed version. Review the change,
update the packed bytes and manifest hashes together, archive the new original
under a new immutable source identity, and remove the expanded copy again.
Do not suppress this conflict check or replace old provenance receipts.

New datasets, checkpoints, ONNX models, raw logs, executed notebooks and report
payloads belong on the data volume. Keep source, small immutable metadata and
the bounded publication data needed by offline CI in Git. A future migration
of raw study receipts needs a portable authenticated artifact store and changes
to all receipt consumers; symlinks to a private host would break fresh clones.

## Private host cleanup archives

The restore CLI also accepts `--manifest` and `--packed-root` for a separately
owned cleanup archive. Entries retain relative `path` and `packed_path`, original
byte counts and SHA-256 hashes, compressed byte counts and hashes, and optional
permission `mode`. Absolute original host paths may be retained as private
metadata, but are never interpreted as restore destinations. Keep private
manifests and logs outside Git and the website publication inputs.

```bash
python scripts/manage_repository_artifacts.py verify \
  --manifest /data/cleanup/archive-manifest.json --packed-root /data/cleanup
python scripts/manage_repository_artifacts.py restore \
  --manifest /data/cleanup/archive-manifest.json --packed-root /data/cleanup \
  --output-dir /data/cleanup/restored
```

Unset `HYPERTAGGING_DATA_ROOT` for packed-only private archives; an explicitly
configured data root continues to require the external-original contract.
Verification checks every selected entry before restoration begins and rechecks
each payload at use time. Memory use is bounded by one expanded artifact rather
than the sum of all artifacts. Identical existing destinations are accepted;
different bytes and escaping paths are rejected. Optional permission restoration
keeps only ordinary permission bits, never setuid/setgid bits.

For host migrations, record active consumers, ownership and scheduler state;
exclude live trees and append-only logs. Copy to a unique data-volume destination,
flush and verify complete hashes and sizes before replacing an untracked legacy
path with a compatibility symlink. Preserve a per-file transaction ledger and
rollback instructions. Never put a private absolute symlink into tracked source.
Regenerable inactive caches may be removed only after recording their recovery
route. Lossless log archives and searchable summaries supplement immutable raw
evidence; missing metrics and infrastructure failures are not scientific zeros.
Filesystem free space does not establish an account quota. Keep transfers bounded
and temporary files on the task-owned data volume.
