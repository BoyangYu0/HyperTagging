#!/usr/bin/env bash
set -euo pipefail

test_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${test_dir}/.." && pwd)"
bundle_dir="$(mktemp -d /tmp/hypertagging-onnx-smoke.XXXXXX)"

cleanup() {
  case "${bundle_dir}" in
    /tmp/hypertagging-onnx-smoke.*) rm -r -- "${bundle_dir}" ;;
  esac
}
trap cleanup EXIT

PYTHONPATH="${project_dir}/src:${project_dir}" \
  "${project_dir}/.venv/bin/python" \
  "${test_dir}/helpers/toy_onnx_bundle.py" "${bundle_dir}"

set +u
source /cvmfs/belle.cern.ch/tools/b2setup light-2607-kasei
set -u

export HYPERTAGGING_ONNX_BUNDLE="${bundle_dir}/manifest.json"
export HYPERTAGGING_SMOKE_MDST="/cvmfs/belle.cern.ch/el9/releases/light-2607-kasei/analysis/tests/mdst.root"
export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
basf2 "${test_dir}/basf2_onnx_full_decay_smoke.py"
basf2 "${test_dir}/basf2_onnx_real_mdst_smoke.py"
basf2 "${test_dir}/basf2_onnx_real_track_smoke.py"
