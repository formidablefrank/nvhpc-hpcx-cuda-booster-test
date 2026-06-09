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
  dist_matmul_stdpar.f90 -o dist_matmul_stdpar.x \
  -lhdf5_fortran -lhdf5

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

run_case() {
  local mode="$1"
  local nodes="$2"
  local ranks="$3"
  local output_file="${FAST}/franco/tmp/C_dist_stdpar_${mode}_${nodes}nodes_${ranks}ranks.h5"

  echo "=== Scaling run: mode=${mode} nodes=${nodes} ranks=${ranks} output=${output_file} ==="
  if [[ "${mode}" == "tuned" ]]; then
    # Accelerator configuration
    export ACC_DEVICE_TYPE=nvidia
    export ACC_DEVICE_NUM=0

    # MPI and PMIX configuration
    export OMPI_MCA_pml="${OMPI_MCA_pml:-ucx}"
    export OMPI_MCA_osc="${OMPI_MCA_osc:-ucx}"
    export PMIX_MCA_gds="${PMIX_MCA_gds:-hash}"

    # NUMA and memory binding
    # place 4 ranks per node and give each rank 8 processing elements
    # then binds those ranks to cores
    tuned_mpirun_args=(--bind-to core --map-by ppr:4:node:PE=8 numactl --localalloc)
    if [[ "${REPORT_BINDINGS:-0}" == "1" ]]; then
      tuned_mpirun_args=(--report-bindings "${tuned_mpirun_args[@]}")
    fi

    # GPU and NIC binding based on local rank
    local_rank="${OMPI_COMM_WORLD_LOCAL_RANK}"
    case $(( local_rank )) in
        0) export CUDA_VISIBLE_DEVICES=0; ucx_net_device=mlx5_0:1 ;;
        1) export CUDA_VISIBLE_DEVICES=1; ucx_net_device=mlx5_1:1 ;;
        2) export CUDA_VISIBLE_DEVICES=2; ucx_net_device=mlx5_2:1 ;;
        3) export CUDA_VISIBLE_DEVICES=3; ucx_net_device=mlx5_3:1 ;;
    esac

    # NIC binding and UCX configuration
    export UCX_LOG_LEVEL="${UCX_LOG_LEVEL:-warn}"
    export UCX_RNDV_THRESH=${UCX_RNDV_THRESH:-8192}
    export UCX_RNDV_FRAG_MEM_TYPE=${UCX_RNDV_FRAG_MEM_TYPE:-cuda}
    export UCX_RNDV_FRAG_SIZE=${UCX_RNDV_FRAG_SIZE:-cuda:32M}
    export UCX_TLS="${UCX_TLS:-rc,cuda_copy,cuda_ipc,sm,self}"
    export UCX_NET_DEVICES="${ucx_net_device}"
    echo "Mode tuned: local rank ${local_rank} using GPU ${CUDA_VISIBLE_DEVICES} and ${UCX_NET_DEVICES}"
    
    MATMUL_BINDER_MODE=tuned \
      "${HPCX_MPI_HOME}/bin/mpirun" -np "${ranks}"  "${tuned_mpirun_args[@]}" \
        "${repo_root}/dist_matmul_stdpar.x" \
        "${output_file}"
  else
    echo "Mode baseline: local rank ${local_rank} using GPU ${CUDA_VISIBLE_DEVICES}"

    MATMUL_BINDER_MODE=baseline \
    "${HPCX_MPI_HOME}/bin/mpirun" -np "${ranks}" \
      "${repo_root}/dist_matmul_stdpar.x" \
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
