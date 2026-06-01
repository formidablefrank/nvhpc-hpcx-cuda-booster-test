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

export SPACK_USER_CONFIG_PATH="${SPACK_USER_CACHE_PATH}"
export SPACK_USER_CACHE_PATH="${SPACK_USER_CACHE_PATH}"

module use /leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/nvhpc-25.11-ayzlbce6ohpmv72ncfv4ogitmppp2usq/modulefiles
module load nvhpc-hpcx-2.20-cuda12/25.11
export HPCX_MPI_HOME="${HPCX_HOME}/ompi"

export CUDA_HOME="$(prefix_for "cuda@12.2.2")"
export NVHPC_CUDA_HOME="${CUDA_HOME}"
export NVCOMPILER_CUDA_HOME="${CUDA_HOME}"
export NVHPC_HOME="$(prefix_for "nvhpc@25.11 ~mpi +blas +lapack default_cuda=12.2")"
export NCCL_HOME="$(prefix_for "nccl@2.22.3-1 +cuda cuda_arch=80 %nvhpc@25.11 ^cuda@12.2.2")"
export CUDNN_HOME="$(prefix_for "cudnn@9.2.0.82-12 %nvhpc@25.11 ^cuda@12.2.2")"
export HDF5_HOME="$(prefix_for "hdf5@1.14.3 +mpi +fortran +hl %nvhpc@25.11 ^hpcx-mpi@2.20")"
export PNETCDF_HOME="$(prefix_for "parallel-netcdf@1.12.3 +cxx +fortran %nvhpc@25.11 ^hpcx-mpi@2.20")"
export NETCDF_C_HOME="$(prefix_for "netcdf-c@4.9.2 +mpi ~blosc ~szip ~zstd %nvhpc@25.11 ^hdf5@1.14.3 +mpi +fortran +hl %nvhpc@25.11 ^hpcx-mpi@2.20")"
export NETCDF_FORTRAN_HOME="$(prefix_for "netcdf-fortran@4.6.1 %nvhpc@25.11 ^netcdf-c@4.9.2 +mpi ~blosc ~szip ~zstd %nvhpc@25.11 ^hdf5@1.14.3 +mpi +fortran +hl %nvhpc@25.11 ^hpcx-mpi@2.20")"

add_prefix "cuda@12.2.2"
add_prefix "nvhpc@25.11 ~mpi +blas +lapack default_cuda=12.2"
add_prefix "nccl@2.22.3-1 +cuda cuda_arch=80 %nvhpc@25.11 ^cuda@12.2.2"
add_prefix "cudnn@9.2.0.82-12 %nvhpc@25.11 ^cuda@12.2.2"
add_prefix "hdf5@1.14.3 +mpi +fortran +hl %nvhpc@25.11 ^hpcx-mpi@2.20"
add_prefix "parallel-netcdf@1.12.3 +cxx +fortran %nvhpc@25.11 ^hpcx-mpi@2.20"
add_prefix "netcdf-c@4.9.2 +mpi ~blosc ~szip ~zstd %nvhpc@25.11 ^hdf5@1.14.3 +mpi +fortran +hl %nvhpc@25.11 ^hpcx-mpi@2.20"
add_prefix "netcdf-fortran@4.6.1 %nvhpc@25.11 ^netcdf-c@4.9.2 +mpi ~blosc ~szip ~zstd %nvhpc@25.11 ^hdf5@1.14.3 +mpi +fortran +hl %nvhpc@25.11 ^hpcx-mpi@2.20"

export CUDAARCHS=80

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_DEBUG_FILENAME="logs/hpcx220-${SLURM_JOB_ID:-nojid}-${world_size}ranks-r${global_rank}.log"
# export NCCL_TOPO_DUMP_FILE="${repo_root}/topo.xml"