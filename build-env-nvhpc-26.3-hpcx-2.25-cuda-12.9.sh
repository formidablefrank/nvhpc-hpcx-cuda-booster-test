#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_dir="${repo_root}/env-nvhpc-26.3-hpcx-2.25-cuda-12.9"
share_root="/leonardo_work/ICT26_MHPC_0/franco"
spack_user_cache_path="${share_root}/spack-user-cache"
spack_user_config_path="${share_root}/spack-user-config"

export SPACK_USER_CACHE_PATH="${spack_user_cache_path}"
export SPACK_USER_CONFIG_PATH="${spack_user_config_path}"
export TMPDIR="${TMPDIR:-${share_root}/tmp}"
unset DEBUG

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

nvhpc_prefix="/leonardo_work/ICT26_MHPC_0/franco/spack-install/env-nvhpc-26.3-hpcx-2.20/linux-rhel8-icelake/gcc-12.2.0/nvhpc-26.3-e2lv6bo2xk2zpw4hoilvgnpbxw3d6zyr"
binutils_spec="binutils@2.42 %gcc@12.2.0"

spack compiler find "${nvhpc_prefix}/Linux_x86_64/26.3/compilers/bin"

spack -e "${env_dir}" concretize -f
spack -e "${env_dir}" install --fail-fast --no-checksum --show-log-on-error "${binutils_spec}"
binutils_prefix="$(spack -e "${env_dir}" location -i "${binutils_spec}")"

python3 - "${SPACK_USER_CONFIG_PATH}/linux/compilers.yaml" "${binutils_prefix}/bin" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
bin_path = sys.argv[2]
lines = path.read_text().splitlines()
out = []
i = 0
while i < len(lines):
    if lines[i].strip() == "spec: nvhpc@=26.3":
        out.append(lines[i])
        i += 1
        while i < len(lines):
            if lines[i].startswith("- compiler:"):
                break
            if lines[i].strip() == "environment: {}":
                out.append("    environment:")
                out.append("      prepend_path:")
                out.append(f"        PATH: {bin_path}")
                i += 1
                continue
            if lines[i].strip().startswith("PATH:") and any(
                line.strip() == "prepend_path:" for line in out[-3:]
            ):
                out.append(f"        PATH: {bin_path}")
                i += 1
                continue
            out.append(lines[i])
            i += 1
        continue
    out.append(lines[i])
    i += 1
path.write_text("\n".join(out) + "\n")
PY
spack clean -m

spack -e "${env_dir}" concretize -f
spack -e "${env_dir}" install --fail-fast --no-checksum --show-log-on-error
