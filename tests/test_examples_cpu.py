import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("script", "example", "contract"),
    [
        ("examples/toy_mc_minimal/run_example.py", "toy_mc_minimal", "toy_mc"),
        ("examples/grafei_minimal/run_example.py", "grafei_minimal", "grafei_combined"),
        ("examples/gpt_like_minimal/run_example.py", "gpt_like_minimal", "gpt_reconstruction_flattened"),
    ],
)
def test_minimal_examples_run_on_cpu(script, example, contract):
    completed = subprocess.run(
        [sys.executable, script],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["example"] == example
    assert payload["contract"] == contract
    assert "preprocess_dry_run" in payload
    assert payload["preprocess_dry_run"]


def test_example_outputs_document_local_data_roots():
    roots = {}
    for script in [
        "examples/toy_mc_minimal/run_example.py",
        "examples/grafei_minimal/run_example.py",
        "examples/gpt_like_minimal/run_example.py",
    ]:
        completed = subprocess.run(
            [sys.executable, script],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        roots[payload["example"]] = payload["input_root"]

    assert roots["toy_mc_minimal"] == "/home/boyang/data/MC"
    assert roots["grafei_minimal"] == "/home/boyang/data/graFEI"
    assert roots["gpt_like_minimal"] == "/home/boyang/data/graFEI"
