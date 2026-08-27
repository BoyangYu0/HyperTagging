# Joint terminal evidence verifier v13 — auth-false v3

The v1/v2 implementation lineage remains preserved and blocked. The successor is sealed at commit `462944635d7b5e1426450650516c298ab43da869`, tag `ht-joint-terminal-evidence-verifier-v13-implementation-20260827-v3`, against spec commit `b1d75b73ad432bb176f9c1bf407c40a6d76850f7` and JSON SHA-256 `63c230481a07408b9be68192e1ea3c2989f027906bd05d64a8abcd82c0ea3583`.

The production surface exposes only `verify_chain` and `decision`. `verify_chain` requires the repository, exactly eight file-backed node ArtifactRefs in terminal→pair→locator→pilot→ladder→pointer→HPO→final order, and trusted roots. It issues a private frozen capability. `decision` consumes only a live, issued, same-process, unexpired capability once; mappings, reconstructed objects, stale instances and cross-process use reject.

Artifact reopening uses normalized trusted-root paths, `O_NOFOLLOW`, lstat/open/fstat identity binding before and after read, device/inode/size/mtime stability, exact byte length and SHA-256, duplicate-key rejecting JSON, exact schema keys/required values, all-false authorization, recursive ArtifactRef reopening, and previous-digest closure across all eight nodes. The only decision remains `E_FEASIBILITY_SHAPE_AUTHORITY`.

Exact hashes: core `ea949df496ddeb0896b5cd7a65dd292ff3fa503fb2d7b58db9634f07ec5f2d61`; router `60ce1959e9122fe47efe06625773dfff86c6b2ec4664c885d0b7061f017a38a8`; public API `260c6273230eb5ce2d111f2a9e0162dac3eec4ff2d5017a3187e481ad0654d28`; v3 tests `462078f08e0ee962ac19a4812f7abcf11a1846934f5777580a2e72304f9fa1e0`; `uv.lock` `7a18fbd4feed4371fa8e8a740f87720462d58c3a8e283402870f375ab744ad18`.

`uv run --frozen --no-sync` results: 198 v3 tests passed with T001–T194 as named parameter IDs and no skips/xfails; combined v13 suites passed 203 tests. Full collection reports 772 tests plus exactly two pre-existing `nbformat` import errors, reconciling the prior 574-test collection with 198 v3 tests.

All implementation, execution, feasibility, scheduler, payload, scientific and submission flags remain false. No dependency install, payload access, GPU operation, scientific training or Slurm submission occurred.
