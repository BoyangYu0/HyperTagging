# Inert 16-key training-selection repromotion

This package is a metadata-only bridge from the historical 15-key
`train_035k` selection to the exact 16-key loader contract. It is based on
commit `43dd53d9f1f93fa7fd942c3bbec411cf54181e91`, tree
`622c2fd37fe9429cb2d8b6f1dfa412c3010ffa97`, and annotated tag
`ht-full-decay-data-loading-hardening-code-only-20260831-v1` (tag object
`fe6788cb5a5935ab900ae025ab353359083c886f`).

The renderer opens exactly four authenticated JSON inputs: inventory, source
roles, the legacy selection, and a byte-identical tracked metadata snapshot of
the historical complete-only index (file hash `f8e09763...8a0bc`). It obtains
`campaign_config_digest` only through a unique `inventory_entry_hash` lookup
in the authenticated inventory, while cross-checking every legacy projected
field and the source-role/task binding. It never opens or resolves raw inputs,
Parquet files, sidecars, completion markers, checkpoints, or the excluded test
role.

The deterministic outputs are:

- selection internal hash `9274e6412a25e8e900efca2c370565346c6d03926e845d7f63a2304a68e87ed4`
  and file hash `d0516ee4db09d7610a614475057af40282594882e03df1e9106e371a6710a660`
  at exactly 33,600 bytes;
- selection fingerprint `75ede96314bf98ff3031cd7ad16aaafcb580370a9ec6b81811cd426d8848ef27`;
- index internal hash `45419bb2d4d70f5344072d70e0f7241d6f235d926ef5bf5dc013ebac9569295e`
  and file hash `8d486304ba8018e03d4126271be24b5b43b15dd2c71f0e5dc8c847071b728097`
  at exactly 768,146 bytes.

Publication is inert and one-shot. A clean annotated implementation tag whose
commit descends from the code-only tag is mandatory. The exact output root and
its sibling claim must both be absent. The renderer exclusively creates the
claim and root; writes each object to a hidden O_EXCL/O_NOFOLLOW staging file;
fsyncs and validates it; promotes it to its SHA-addressed name with Linux
`renameat2(RENAME_NOREPLACE)`; writes the receipt last; seals ordinary files
read-only with one link; fsyncs each boundary; and never updates a live pointer.
A collision or partial publication permanently blocks reuse of that namespace.

The exact closed six-key authorization set covers execution, science,
submission, Slurm actions, source-payload access, and live-pointer publication;
all six values are false. The committed package is not a runtime activation.
Independent audit, consumer hash pinning, immutable
promotion approval, and clean-commit smoke/provenance evidence remain external
gates.
