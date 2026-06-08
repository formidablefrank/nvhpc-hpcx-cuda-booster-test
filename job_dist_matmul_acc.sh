#!/bin/bash
#SBATCH --job-name=dist-matmul-acc
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
#SBATCH --output=logs/slurm-matmul-%j.out
#SBATCH --error=logs/slurm-matmul-%j.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${repo_root}"

mkdir -p logs
source "${repo_root}/hpcx-only-env.sh"

echo "Compiling..."
"${HPCX_MPI_HOME}/bin/mpif90" -O3 -acc -gpu=cc80 -Minfo=accel \
  -I"${HDF5_HOME}/include" \
  -L"${HDF5_HOME}/lib" \
  dist_matmul_acc.f90 -o dist_matmul_acc.x \
  -lhdf5_fortran -lhdf5

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

run_case() {
  local mode="$1"
  local nodes="$2"
  local ranks="$3"
  local output_file="${FAST}/franco/tmp/C_dist_acc_${mode}_${nodes}nodes_${ranks}ranks.h5"

  echo "=== Scaling run: mode=${mode} nodes=${nodes} ranks=${ranks} output=${output_file} ==="
  if [[ "${mode}" == "tuned" ]]; then
    export OMPI_MCA_pml="${OMPI_MCA_pml:-ucx}"
    export OMPI_MCA_osc="${OMPI_MCA_osc:-ucx}"
    export PMIX_MCA_gds="${PMIX_MCA_gds:-hash}"

    # place 4 ranks per node and give each rank 8 processing elements
    # then binds those ranks to cores 
    tuned_mpirun_args=(--bind-to core --map-by ppr:4:node:PE=8 numactl --localalloc)
    if [[ "${REPORT_BINDINGS:-0}" == "1" ]]; then
      tuned_mpirun_args=(--report-bindings "${tuned_mpirun_args[@]}")
    fi
    
    MATMUL_BINDER_MODE=tuned \
      "${HPCX_MPI_HOME}/bin/mpirun" -np "${ranks}"  "${tuned_mpirun_args[@]}" \
        "${repo_root}/binder.sh" \
        "${repo_root}/dist_matmul_acc.x" \
        "${output_file}"
  else
    MATMUL_BINDER_MODE=baseline \
    "${HPCX_MPI_HOME}/bin/mpirun" -np "${ranks}" \
      "${repo_root}/binder.sh" \
      "${repo_root}/dist_matmul_acc.x" \
      "${output_file}"
  fi
}

echo "Running scaling analysis: baseline defaults vs tuned UCX/MPI/NUMA mapping..."
tasks_per_node="${SLURM_NTASKS_PER_NODE:-4}"
for nodes in 1 2 4 8; do
  ranks=$((nodes * tasks_per_node))
  run_case baseline "${nodes}" "${ranks}"
  run_case tuned "${nodes}" "${ranks}"
done
