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

mkdir -p \
  "${SPACK_USER_CACHE_PATH}" \
  "${SPACK_USER_CONFIG_PATH}" \
  "${TMPDIR}" \
  "${share_root}/spack-source-cache" \
  "${share_root}/spack-binary-index" \
  "${share_root}/spack-bootstrap"

prepend_path() {
  local var_name="$1"
  local value="$2"

  if [[ -z "${value}" || ! -e "${value}" ]]; then
    return
  fi

  if [[ -n "${!var_name:-}" ]]; then
    export "${var_name}=${value}:${!var_name}"
  else
    export "${var_name}=${value}"
  fi
}

add_prefix() {
  local spec="$1"
  local prefix

  prefix="$(spack -e "${env_dir}" location -i "$spec")"
  prepend_path CMAKE_PREFIX_PATH "${prefix}"

  if [[ -d "${prefix}/bin" ]]; then
    prepend_path PATH "${prefix}/bin"
  fi
  if [[ -d "${prefix}/include" ]]; then
    prepend_path CPATH "${prefix}/include"
    prepend_path C_INCLUDE_PATH "${prefix}/include"
    prepend_path CPLUS_INCLUDE_PATH "${prefix}/include"
  fi
  if [[ -d "${prefix}/lib" ]]; then
    prepend_path LIBRARY_PATH "${prefix}/lib"
    prepend_path LD_LIBRARY_PATH "${prefix}/lib"
    prepend_path PKG_CONFIG_PATH "${prefix}/lib/pkgconfig"
  fi
  if [[ -d "${prefix}/lib64" ]]; then
    prepend_path LIBRARY_PATH "${prefix}/lib64"
    prepend_path LD_LIBRARY_PATH "${prefix}/lib64"
    prepend_path PKG_CONFIG_PATH "${prefix}/lib64/pkgconfig"
  fi
}

prefix_for() {
  spack -e "${env_dir}" location -i "$1"
}

set +u
module purge
module load spack
set -u

export SPACK_USER_CACHE_PATH="${spack_user_cache_path}"
export SPACK_USER_CONFIG_PATH="${spack_user_config_path}"

spack bootstrap root "${share_root}/spack-bootstrap" >/dev/null

export CUDA_HOME="$(prefix_for "cuda@12.9 %nvhpc@26.3")"
export NVHPC_CUDA_HOME="${CUDA_HOME}"
export NVCOMPILER_CUDA_HOME="${CUDA_HOME}"
export NVHPC_HOME="$(prefix_for "nvhpc@26.3 %gcc@12.2.0")"
export NVHPC_VERSION_HOME="${NVHPC_HOME}/Linux_x86_64/26.3"
export HPCX_MPI_HOME="$(prefix_for "hpcx-mpi@2.25.1")"
export HPCX_HOME="${HPCX_MPI_HOME%/ompi}"
export BINUTILS_HOME="$(prefix_for "binutils@2.42 %gcc@12.2.0")"
export NCCL_HOME="$(prefix_for "nccl@2.22.3-1 %nvhpc@26.3 cuda_arch=80 ^cuda@12.9")"
export GDRCOPY_HOME="$(prefix_for "gdrcopy@2.5.1 %nvhpc@26.3 +cuda cuda_arch=80 ^cuda@12.9")"
export CUDNN_HOME="$(prefix_for "cudnn@9.23.0.39-12 %nvhpc@26.3 ^cuda@12.9")"
export HDF5_HOME="$(prefix_for "hdf5@1.14.3 %nvhpc@26.3")"
export PNETCDF_HOME="$(prefix_for "parallel-netcdf@1.12.3 %nvhpc@26.3")"
export NETCDF_C_HOME="$(prefix_for "netcdf-c@4.9.2 %nvhpc@26.3")"
export NETCDF_FORTRAN_HOME="$(prefix_for "netcdf-fortran@4.6.1 %nvhpc@26.3")"

prepend_path PATH "${NVHPC_VERSION_HOME}/compilers/bin"
prepend_path MANPATH "${NVHPC_VERSION_HOME}/compilers/man"
prepend_path LIBRARY_PATH "${NVHPC_VERSION_HOME}/compilers/lib"
prepend_path LD_LIBRARY_PATH "${NVHPC_VERSION_HOME}/compilers/lib"

export CC="${NVHPC_VERSION_HOME}/compilers/bin/nvc"
export CXX="${NVHPC_VERSION_HOME}/compilers/bin/nvc++"
export F77="${NVHPC_VERSION_HOME}/compilers/bin/nvfortran"
export FC="${NVHPC_VERSION_HOME}/compilers/bin/nvfortran"

add_prefix "cuda@12.9 %nvhpc@26.3"
add_prefix "hpcx-mpi@2.25.1"
add_prefix "binutils@2.42 %gcc@12.2.0"
add_prefix "nccl@2.22.3-1 %nvhpc@26.3 cuda_arch=80 ^cuda@12.9"
add_prefix "gdrcopy@2.5.1 %nvhpc@26.3 +cuda cuda_arch=80 ^cuda@12.9"
add_prefix "cudnn@9.23.0.39-12 %nvhpc@26.3 ^cuda@12.9"
add_prefix "hdf5@1.14.3 %nvhpc@26.3"
add_prefix "parallel-netcdf@1.12.3 %nvhpc@26.3"
add_prefix "netcdf-c@4.9.2 %nvhpc@26.3"
add_prefix "netcdf-fortran@4.6.1 %nvhpc@26.3"

export CUDAARCHS=80
