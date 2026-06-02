#!/bin/bash
#SBATCH --job-name=dist-matmul-stdpar
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=490000MB
##SBATCH --mem-bind=local             equivalent to numactl --localalloc in the job steps below
##SBATCH --distribution=block:block   equivalent to --map-by ppr:4:node in the mpirun below
#SBATCH --exclusive
#SBATCH --account=ICT26_MHPC_0
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/slurm-stdpar-matmul-%j.out
#SBATCH --error=logs/slurm-stdpar-matmul-%j.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${repo_root}"

mkdir -p logs
source "${repo_root}/hpcx-only-env.sh"

echo "Compiling stdpar variant..."
"${HPCX_MPI_HOME}/bin/mpif90" -O3 -stdpar=gpu -gpu=cc80,mem:managed -Minfo=stdpar \
  -I"${HDF5_HOME}/include" \
  -L"${HDF5_HOME}/lib" \
  dist_matmul_stdpar.f90 -o dist_matmul_stdpar.x \
  -lhdf5_fortran -lhdf5

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export UCX_LOG_LEVEL="${UCX_LOG_LEVEL:-warn}"
export UCX_TLS="${UCX_TLS:-rc,cuda_copy,cuda_ipc,sm,self}"
export OMPI_MCA_pml="${OMPI_MCA_pml:-ucx}"
export OMPI_MCA_osc="${OMPI_MCA_osc:-ucx}"
export PMIX_MCA_gds="${PMIX_MCA_gds:-hash}"
export UCX_RNDV_THRESH="${UCX_RNDV_THRESH:-8192}"

tuned_mpirun_args=(--bind-to core --map-by ppr:4:node:PE=8)
if [[ "${REPORT_BINDINGS:-0}" == "1" ]]; then
  tuned_mpirun_args=(--report-bindings "${tuned_mpirun_args[@]}")
fi

run_case() {
  local mode="$1"
  local nodes="$2"
  local ranks="$3"
  local output_file="C_dist_stdpar_${mode}_${nodes}nodes_${ranks}ranks.h5"

  echo "=== Stdpar scaling run: mode=${mode} nodes=${nodes} ranks=${ranks} output=${output_file} ==="
  if [[ "${mode}" == "tuned" ]]; then
    MATMUL_BINDER_MODE=tuned \
      "${HPCX_MPI_HOME}/bin/mpirun" -np "${ranks}" "${tuned_mpirun_args[@]}" \
        numactl --localalloc \
        "${repo_root}/binder.sh" \
        "${repo_root}/dist_matmul_stdpar.x" \
        "${output_file}"
  else
    env -u UCX_TLS -u UCX_NET_DEVICES -u OMPI_MCA_pml -u OMPI_MCA_osc -u PMIX_MCA_gds -u UCX_RNDV_THRESH \
      MATMUL_BINDER_MODE=baseline \
      "${HPCX_MPI_HOME}/bin/mpirun" -np "${ranks}" \
        "${repo_root}/binder.sh" \
        "${repo_root}/dist_matmul_stdpar.x" \
        "${output_file}"
  fi
}

echo "Running stdpar scaling analysis: baseline defaults vs tuned UCX/MPI/NUMA mapping..."
tasks_per_node="${SLURM_NTASKS_PER_NODE:-4}"
for nodes in 1 2 4 8; do
  ranks=$((nodes * tasks_per_node))
  run_case baseline "${nodes}" "${ranks}"
  run_case tuned "${nodes}" "${ranks}"
done
