# GraFEI Minimal Example

Runs CPU-only GraFEI fixtures through the migrated data contract and the
validated reconstruction dry-run path. It also prints the dry-run command for
the legacy GraFEI preprocessing adapter.

Original GraFEI inputs are expected under `/home/boyang/data/graFEI` before
preprocessing. This example does not read that directory.

```bash
uv --cache-dir /tmp/uv-cache run python examples/grafei_minimal/run_example.py
```
