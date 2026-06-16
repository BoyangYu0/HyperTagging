# Toy-MC Minimal Example

Runs a CPU-only Toy-MC fixture through the migrated data contract and a tiny
embedding-loss calculation. It also prints the dry-run command for the legacy
Toy-MC preprocessing adapter.

Full Toy-MC inputs are expected under `/home/boyang/data/MC` after BASF2
generation and before preprocessing. This example does not read that directory.

```bash
uv --cache-dir /tmp/uv-cache run python examples/toy_mc_minimal/run_example.py
```
