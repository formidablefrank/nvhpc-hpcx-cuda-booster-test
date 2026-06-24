#!/bin/bash
set -euo pipefail

local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:?OMPI_COMM_WORLD_LOCAL_RANK is not set}"
global_rank="${OMPI_COMM_WORLD_RANK:-unknown}"

case "${local_rank}" in
  0) gpu=0; ucx_net_device=mlx5_0:1 ;;
  1) gpu=1; ucx_net_device=mlx5_1:1 ;;
  2) gpu=2; ucx_net_device=mlx5_2:1 ;;
  3) gpu=3; ucx_net_device=mlx5_3:1 ;;
  *)
    echo "[gpu_nic_ucx] ERROR: local rank ${local_rank} is invalid for 4 GPUs/node. Use --ntasks-per-node=4." >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="${gpu}"
export NVCOMPILER_ACC_DEVICE_TYPE=nvidia
export NVCOMPILER_ACC_DEVICE_NUM=0
export UCX_TLS="${UCX_TLS:-rc,cuda_copy,cuda_ipc,sm,self}"
export UCX_NET_DEVICES="${ucx_net_device}"
export UCX_LOG_LEVEL="${UCX_LOG_LEVEL:-warn}"
# export UCX_RNDV_THRESH=${UCX_RNDV_THRESH:-8192}
# export UCX_RNDV_FRAG_MEM_TYPE=${UCX_RNDV_FRAG_MEM_TYPE:-cuda}
# export UCX_RNDV_FRAG_SIZE=${UCX_RNDV_FRAG_SIZE:-cuda:32M}

echo "[gpu_nic_ucx] host=$(hostname) global=${global_rank} local=${local_rank} gpu=${CUDA_VISIBLE_DEVICES} hca=${UCX_NET_DEVICES} UCX_TLS=${UCX_TLS}"
exec "$@"
