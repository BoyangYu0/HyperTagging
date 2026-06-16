# GPT-Like Minimal Example

Runs CPU-only GPT-like/autoregressive fixtures through the migrated collate
helper and combined `MultiGPT` dry-run path.

Original GraFEI inputs are expected under `/home/boyang/data/graFEI` before
preprocessing. This example does not read that directory.

```bash
uv --cache-dir /tmp/uv-cache run python examples/gpt_like_minimal/run_example.py
```
