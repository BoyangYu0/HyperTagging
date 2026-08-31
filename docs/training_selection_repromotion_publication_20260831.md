# One-use metadata-package publication boundary

The sealed repromotion package remains inert. Its default authorization map is
closed and entirely false. A package may be materialized only through the
separate publish-once helper and an external, mode-0444, self-hashed capability
whose exact metadata-publication flag alone is true. The command and capability
must both be consumed by root; non-root execution or a non-root-owned
capability is rejected.

The capability binds one clean annotated implementation tag, repository and
contract hashes, all four authenticated metadata inputs, one absolute fresh
namespace, deterministic output hashes and byte sizes, the expected package
receipt hash, the complete command/environment, and the exact `env`, Git, and
Python executable closure. It expires, is one-use, and never authorizes retry,
consumer pinning, pointer activation, payload access, science, submission, or
Slurm.

After all read-only checks and an in-memory deterministic build, the helper
creates a sibling authorization lock with O_EXCL/O_NOFOLLOW. Any failure or
uncertainty after that lock consumes the namespace permanently. The existing
publisher then creates its own claim, hidden staged CAS objects, and package
receipt. Only after rehashing and checking modes, links, sizes, and the package
receipt does the helper write a sibling accepted-publication receipt. A package
without that accepted receipt is ineligible for any consumer pin and must not
be retried or repaired without a separately implemented and audited
reconciliation path.

Publication is not activation. It changes no tracked configuration or live
pointer and grants no consumer or scientific authority. A later consumer-pin
commit must independently bind the accepted receipt plus the exact selection
and index CAS paths, file/internal hashes, sizes, and train/validation-only role
contract. Any live pointer, if ever needed, is a third separately authorized
Git compare-and-swap operation; direct CAS paths are preferred.

No real authorization file is embedded in the repository. Tests mint only
temporary capabilities and temporary output namespaces.
