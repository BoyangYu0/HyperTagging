"""Both compatibility layouts must finish, and either failure blocks publication."""
import os
from pathlib import Path
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('text_exit,basf2_exit', [(0, 0), (7, 0), (0, 9), (7, 9)])
def test_layout_step_propagates_each_failure(tmp_path, text_exit, basf2_exit):
    workflow = yaml.safe_load((ROOT / '.github/workflows/docs.yml').read_text())
    step = next(step for step in workflow['jobs']['build']['steps']
                if step.get('name') == 'Validate text and basf2 discovery layouts')
    fake_python = tmp_path / 'python'
    fake_python.write_text('''#!/bin/bash
case "$*" in
  *"--builder text"*) touch "$RUNNER_TEMP/text-finished"; exit "$TEXT_EXIT" ;;
  *"--layout basf2"*) touch "$RUNNER_TEMP/basf2-finished"; exit "$BASF2_EXIT" ;;
  *) exit 97 ;;
esac
''')
    fake_python.chmod(0o755)
    env = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
           'RUNNER_TEMP': str(tmp_path), 'TEXT_EXIT': str(text_exit),
           'BASF2_EXIT': str(basf2_exit)}
    result = subprocess.run(['bash', '--noprofile', '--norc', '-e', '-o', 'pipefail',
                             '-c', step['run']], env=env, capture_output=True, timeout=10)
    assert (result.returncode == 0) == (text_exit == basf2_exit == 0)
    assert (tmp_path / 'text-finished').exists()
    assert (tmp_path / 'basf2-finished').exists()
