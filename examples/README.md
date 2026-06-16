# Examples

These examples are CPU-only smoke runs built from tiny synthetic fixtures and
validated dry-run paths. They do not read full datasets or reproduce physics
performance.

- `toy_mc_minimal/`: validates the Toy-MC contract, computes a tiny embedding
  loss, and prints the legacy Toy-MC preprocessing dry-run command.
- `grafei_minimal/`: validates the GraFEI combined reconstruction contract and
  runs the reconstruction dry-run.
- `gpt_like_minimal/`: validates the GPT-like collate path and runs the combined
  `MultiGPT` dry-run.

Full-data roots documented for later reproduction:

- Toy-MC after BASF2 generation and before preprocessing:
  `/home/boyang/data/MC`
- Original GraFEI before preprocessing:
  `/home/boyang/data/graFEI`

Run all examples individually:

```bash
uv --cache-dir /tmp/uv-cache run python examples/toy_mc_minimal/run_example.py
uv --cache-dir /tmp/uv-cache run python examples/grafei_minimal/run_example.py
uv --cache-dir /tmp/uv-cache run python examples/gpt_like_minimal/run_example.py
```
