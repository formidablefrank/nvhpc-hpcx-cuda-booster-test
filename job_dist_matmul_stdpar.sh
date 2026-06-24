#!/bin/bash
#SBATCH --job-name=dist-matmul-stdpar
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=490000MB
##SBATCH --mem-bind=local             equivalent to numactl --localalloc in the mpirun below
##SBATCH --distribution=block:block   equivalent to --map-by ppr:4:node in the mpirun below
#SBATCH --exclusive
#SBATCH --account=ICT26_MHPC_0
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/slurm-stdpar-matmul-%j.out
#SBATCH --error=logs/slurm-stdpar-matmul-%j.err

set -euo pipefail

mkdir -p logs
# source load-env-nvhpc-26.3-hpcx-2.20.sh
source load-env-nvhpc-26.3-hpcx-2.25-cuda-12.9.sh

run_version_cmd() {
  local label="$1"
  shift

  echo "--- ${label} ---"
  if "$@"; then
    return 0
  fi
  echo "${label}: unavailable or failed"
}

print_software_stack() {
  echo "=== Software stack diagnostics ==="
  echo "Date: $(date --iso-8601=seconds)"
  echo "Host: $(hostname)"
  echo "Working directory: $(pwd)"
  echo "Slurm job: ${SLURM_JOB_ID:-unset}"
  echo "Slurm nodes: ${SLURM_JOB_NUM_NODES:-unset}"
  echo "Slurm tasks per node: ${SLURM_NTASKS_PER_NODE:-unset}"
  echo "Slurm CPUs per task: ${SLURM_CPUS_PER_TASK:-unset}"

  echo "--- Loaded modules ---"
  module list 2>&1 || echo "module list unavailable"

  echo "--- Key paths ---"
  printf 'CUDA_HOME=%s\n' "${CUDA_HOME:-unset}"
  printf 'NVHPC_HOME=%s\n' "${NVHPC_HOME:-unset}"
  printf 'NVHPC_CUDA_HOME=%s\n' "${NVHPC_CUDA_HOME:-unset}"
  printf 'NVCOMPILER_CUDA_HOME=%s\n' "${NVCOMPILER_CUDA_HOME:-unset}"
  printf 'HPCX_HOME=%s\n' "${HPCX_HOME:-unset}"
  printf 'HPCX_MPI_HOME=%s\n' "${HPCX_MPI_HOME:-unset}"
  printf 'NCCL_HOME=%s\n' "${NCCL_HOME:-unset}"
  printf 'GDRCOPY_HOME=%s\n' "${GDRCOPY_HOME:-unset}"
  printf 'CUDNN_HOME=%s\n' "${CUDNN_HOME:-unset}"
  printf 'HDF5_HOME=%s\n' "${HDF5_HOME:-unset}"
  printf 'PNETCDF_HOME=%s\n' "${PNETCDF_HOME:-unset}"
  printf 'NETCDF_C_HOME=%s\n' "${NETCDF_C_HOME:-unset}"
  printf 'NETCDF_FORTRAN_HOME=%s\n' "${NETCDF_FORTRAN_HOME:-unset}"

  if [[ -x "${HPCX_MPI_HOME:-}/bin/mpif90" ]]; then
    run_version_cmd "HPC-X mpif90" "${HPCX_MPI_HOME}/bin/mpif90" --version
  fi
  if [[ -x "${HPCX_MPI_HOME:-}/bin/mpirun" ]]; then
    run_version_cmd "HPC-X mpirun" "${HPCX_MPI_HOME}/bin/mpirun" --version
  fi
  if command -v nvfortran >/dev/null 2>&1; then
    run_version_cmd "NVHPC nvfortran" nvfortran --version
  fi
  if [[ -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
    run_version_cmd "CUDA nvcc" "${CUDA_HOME}/bin/nvcc" --version
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    run_version_cmd "NVIDIA driver/GPU" nvidia-smi
  fi
  if command -v ucx_info >/dev/null 2>&1; then
    run_version_cmd "UCX" ucx_info -v
  elif [[ -x "${HPCX_HOME:-}/ucx/bin/ucx_info" ]]; then
    run_version_cmd "UCX" "${HPCX_HOME}/ucx/bin/ucx_info" -v
  fi
  if [[ -x "${HDF5_HOME:-}/bin/h5pfc" ]]; then
    run_version_cmd "HDF5 h5pfc" "${HDF5_HOME}/bin/h5pfc" -showconfig
  elif [[ -x "${HDF5_HOME:-}/bin/h5cc" ]]; then
    run_version_cmd "HDF5 h5cc" "${HDF5_HOME}/bin/h5cc" -showconfig
  fi
  if [[ -x "${NETCDF_C_HOME:-}/bin/nc-config" ]]; then
    run_version_cmd "NetCDF-C" "${NETCDF_C_HOME}/bin/nc-config" --version
  fi
  if [[ -x "${NETCDF_FORTRAN_HOME:-}/bin/nf-config" ]]; then
    run_version_cmd "NetCDF-Fortran" "${NETCDF_FORTRAN_HOME}/bin/nf-config" --version
  fi
  echo "=== End software stack diagnostics ==="
}

print_software_stack

echo "Compiling..."
"${HPCX_MPI_HOME}/bin/mpif90" -O3 -stdpar=gpu -gpu=cc80,mem:separate -Minfo=stdpar \
  -I"${HDF5_HOME}/include" \
  -L"${HDF5_HOME}/lib" \
  dist_matmul_stdpar.f90 -o dist_matmul_stdpar.x \
  -lhdf5_fortran -lhdf5

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

base_env=(
  env
  -u OMPI_MCA_pml
  -u OMPI_MCA_osc
  -u PMIX_MCA_gds
  -u OMPI_MCA_btl
  -u UCX_NET_DEVICES
  -u UCX_TLS
  -u ACC_DEVICE_TYPE
  -u ACC_DEVICE_NUM
  -u CUDA_VISIBLE_DEVICES
)

gpu_nic_ucx_wrapper="./binder.sh"

run_case() {
  local mode="$1"
  local nodes="$2"
  local ranks="$3"
  local output_file="${FAST}/franco/tmp/C_dist_stdpar_${mode}_${nodes}nodes_${ranks}ranks.h5"
  local pe="${SLURM_CPUS_PER_TASK:-8}"
  local -a env_args=("${base_env[@]}")
  local -a mpirun_args=()
  local -a launcher_args=("./dist_matmul_stdpar.x" "${output_file}")

  echo "=== Scaling run: mode=${mode} nodes=${nodes} ranks=${ranks} output=${output_file} ==="
  case "${mode}" in
    baseline)
      ;;
    mpi_pmix)
      env_args+=(
        OMPI_MCA_pml=ucx
        OMPI_MCA_osc=ucx
        PMIX_MCA_gds=hash
      )
      if [[ "${OMPI_BTL_SET:-1}" == "1" ]]; then
        env_args+=(OMPI_MCA_btl=^openib)
      fi
      ;;
    numa_mem)
      mpirun_args=(--bind-to core --map-by "ppr:4:node:PE=${pe}")
      launcher_args=(numactl --localalloc "${launcher_args[@]}")
      if [[ "${REPORT_BINDINGS:-0}" == "1" ]]; then
        mpirun_args=(--report-bindings "${mpirun_args[@]}")
      fi
      ;;
    gpu_nic_ucx)
      launcher_args=("${gpu_nic_ucx_wrapper}" "${launcher_args[@]}")
      ;;
    all_tuned)
      env_args+=(
        OMPI_MCA_pml=ucx
        OMPI_MCA_osc=ucx
        PMIX_MCA_gds=hash
      )
      if [[ "${OMPI_BTL_SET:-1}" == "1" ]]; then
        env_args+=(OMPI_MCA_btl=^openib)
      fi
      mpirun_args=(--bind-to core --map-by "ppr:4:node:PE=${pe}")
      launcher_args=(numactl --localalloc "${gpu_nic_ucx_wrapper}" "${launcher_args[@]}")
      if [[ "${REPORT_BINDINGS:-0}" == "1" ]]; then
        mpirun_args=(--report-bindings "${mpirun_args[@]}")
      fi
      ;;
    *)
      echo "ERROR: unknown benchmark mode '${mode}'" >&2
      exit 2
      ;;
  esac

  echo "Mode ${mode}: env=${env_args[*]}"
  echo "Mode ${mode}: mpirun_args=${mpirun_args[*]:-(none)}"
  echo "Mode ${mode}: launcher=${launcher_args[*]}"

  "${env_args[@]}" \
    "${HPCX_MPI_HOME}/bin/mpirun" -np "${ranks}" "${mpirun_args[@]}" \
    "${launcher_args[@]}"
}

echo "Running isolated fine-tuning scaling analysis..."
tasks_per_node="${SLURM_NTASKS_PER_NODE:-4}"
for nodes in 1 2 4 8; do
  ranks=$((nodes * tasks_per_node))
  for mode in baseline mpi_pmix numa_mem gpu_nic_ucx all_tuned; do
    run_case "${mode}" "${nodes}" "${ranks}"
  done
done
