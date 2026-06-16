import os
import sys
from pathlib import Path
import subprocess

from hypertagging.data.preprocessing import (
    DEFAULT_GRAFEI_INPUT_ROOT,
    DEFAULT_TOY_MC_INPUT_ROOT,
    PreprocessingPlan,
    assert_legacy_scripts_exist,
    build_gpt_like_plan,
    build_grafei_plan,
    build_toy_mc_dataprod_plan,
    build_toy_mc_preprocess_plan,
    run_legacy_preprocessing,
)
from hypertagging.data.toy_mc import prepare_toy_mc
from hypertagging.data.grafei import prepare_grafei
from hypertagging.data.gpt_like import prepare_gpt_like


def test_phase4_default_input_roots_match_migration_plan():
    assert DEFAULT_TOY_MC_INPUT_ROOT == Path("/home/boyang/data/MC")
    assert DEFAULT_GRAFEI_INPUT_ROOT == Path("/home/boyang/data/graFEI")


def test_toy_mc_preprocess_dry_run_builds_legacy_command():
    result = prepare_toy_mc(7, dry_run=True)

    assert result.plan.kind == "toy_mc_preprocess"
    assert result.plan.input_root == DEFAULT_TOY_MC_INPUT_ROOT
    assert result.command[-2].endswith("HyperTagging/ak/preprocess_ak.py")
    assert result.command[-1] == "7"
    assert "tokenization" in " ".join(result.plan.notes)


def test_toy_mc_dataprod_and_grafei_scripts_exist():
    plans = [
        build_toy_mc_dataprod_plan(0, 1),
        build_toy_mc_preprocess_plan(1),
        build_grafei_plan(0),
        build_grafei_plan(0, reduced=False),
        build_gpt_like_plan(0),
    ]

    assert_legacy_scripts_exist(plans)


def test_grafei_and_gpt_like_dry_runs_use_grafei_input_root():
    grafei = prepare_grafei(3, dry_run=True)
    gpt_like = prepare_gpt_like(4, dry_run=True)

    assert grafei.plan.kind == "grafei"
    assert grafei.plan.input_root == DEFAULT_GRAFEI_INPUT_ROOT
    assert grafei.command[-2].endswith("graFEI_reduced/produce_train_data_grafei.py")
    assert grafei.command[-1] == "3"
    assert gpt_like.plan.kind == "gpt_like"
    assert gpt_like.plan.input_root == DEFAULT_GRAFEI_INPUT_ROOT
    assert gpt_like.command[-2].endswith("graFEI_gpt/produce_train_data_grafei.py")
    assert gpt_like.command[-1] == "4"


def test_adapter_can_pass_roots_to_a_cpu_only_subprocess(tmp_path):
    script = tmp_path / "echo_roots.py"
    script.write_text(
        "import os\n"
        "print(os.environ['HYPERTAGGING_INPUT_ROOT'])\n"
        "print(os.environ['HYPERTAGGING_OUTPUT_ROOT'])\n",
        encoding="utf-8",
    )
    plan = PreprocessingPlan(
        kind="grafei",
        legacy_script=script,
        argv=(),
        input_root=tmp_path / "input",
        output_root=tmp_path / "output",
        cwd=tmp_path,
    )

    result = run_legacy_preprocessing(
        plan,
        dry_run=False,
        python_executable=sys.executable,
    )

    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        os.fspath(tmp_path / "input"),
        os.fspath(tmp_path / "output"),
    ]
