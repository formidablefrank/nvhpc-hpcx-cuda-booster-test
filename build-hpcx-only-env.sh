#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_dir="${repo_root}/env-nvhpc-hpcx"
share_root="/leonardo_work/ICT26_MHPC_0/franco"
spack_user_cache_path="${share_root}/spack-user-cache"
spack_user_config_path="${share_root}/spack-user-config"

export SPACK_USER_CACHE_PATH="${spack_user_cache_path}"
export SPACK_USER_CONFIG_PATH="${spack_user_config_path}"
export TMPDIR="${TMPDIR:-${share_root}/tmp}"

mkdir -p \
  "${SPACK_USER_CACHE_PATH}" \
  "${SPACK_USER_CONFIG_PATH}" \
  "${TMPDIR}" \
  "${share_root}/spack-source-cache" \
  "${share_root}/spack-binary-index" \
  "${share_root}/spack-bootstrap"

set +u
module load spack
set -u

export SPACK_USER_CACHE_PATH="${spack_user_cache_path}"
export SPACK_USER_CONFIG_PATH="${spack_user_config_path}"

spack bootstrap root "${share_root}/spack-bootstrap"

spack -e "${env_dir}" concretize -f
spack -e "${env_dir}" install --fail-fast
