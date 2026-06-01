#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_dir="${repo_root}/env-nvhpc-hpcx"
export SPACK_USER_CACHE_PATH="/leonardo_scratch/large/userexternal/jrayo000/spack-user-cache"
export TMPDIR="${TMPDIR:-/leonardo_scratch/large/userexternal/jrayo000/tmp}"

mkdir -p \
  "${SPACK_USER_CACHE_PATH}" \
  "${TMPDIR}" \
  /leonardo_scratch/large/userexternal/jrayo000/spack-source-cache \
  /leonardo_scratch/large/userexternal/jrayo000/spack-binary-index

set +u
module load spack
set -u

export SPACK_USER_CONFIG_PATH="${SPACK_USER_CACHE_PATH}"
export SPACK_USER_CACHE_PATH="${SPACK_USER_CACHE_PATH}"

spack -e "${env_dir}" concretize -f
spack -e "${env_dir}" install --fail-fast
