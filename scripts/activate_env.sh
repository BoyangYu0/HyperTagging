#!/usr/bin/env bash
# Source this file to activate a HyperTagging uv-managed environment:
#   source scripts/activate_env.sh project
#   source scripts/activate_env.sh gpu

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source this helper instead of executing it" >&2
  exit 64
fi

_ht_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)" || {
  echo "could not resolve the HyperTagging repository" >&2
  return 1
}
_ht_mode="${1:-}"
case "${_ht_mode}" in
  project)
    _ht_environment="${_ht_repo_root}/.venv"
    ;;
  gpu)
    _ht_environment="/project/agkuhr/users/boyang/envs/hypertagging-gpu-cu126-v1"
    ;;
  *)
    echo "usage: source ${BASH_SOURCE[0]} {project|gpu}" >&2
    unset _ht_repo_root _ht_mode
    return 64
    ;;
esac

if [[ ! -f "${_ht_environment}/bin/activate" ]]; then
  echo "HyperTagging ${_ht_mode} environment is absent: ${_ht_environment}" >&2
  unset _ht_repo_root _ht_mode _ht_environment
  return 1
fi

# The standard venv activation script safely replaces any active environment.
# shellcheck disable=SC1090
source "${_ht_environment}/bin/activate"
cd -- "${_ht_repo_root}" || return 1
export HYPERTAGGING_ENV_MODE="${_ht_mode}"
unset _ht_repo_root _ht_mode _ht_environment
