#!/bin/bash
set -euo pipefail

local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:?OMPI_COMM_WORLD_LOCAL_RANK is not set}"
# global_rank="${OMPI_COMM_WORLD_RANK:-unknown}"
# world_size="${OMPI_COMM_WORLD_SIZE:-unknown}"
binder_mode="${MATMUL_BINDER_MODE:-tuned}"

case $(( local_rank )) in
    0) export CUDA_VISIBLE_DEVICES=0; ucx_net_device=mlx5_0:1 ;;
    1) export CUDA_VISIBLE_DEVICES=1; ucx_net_device=mlx5_1:1 ;;
    2) export CUDA_VISIBLE_DEVICES=2; ucx_net_device=mlx5_2:1 ;;
    3) export CUDA_VISIBLE_DEVICES=3; ucx_net_device=mlx5_3:1 ;;
esac

if [[ "${binder_mode}" == "tuned" ]]; then
    export UCX_LOG_LEVEL="${UCX_LOG_LEVEL:-warn}"
    export UCX_RNDV_THRESH="${UCX_RNDV_THRESH:-8192}"
    export UCX_TLS="${UCX_TLS:-rc,cuda_copy,cuda_ipc,sm,self}"
    export UCX_NET_DEVICES="${ucx_net_device}"
    echo "Mode tuned: local rank ${local_rank} using GPU ${CUDA_VISIBLE_DEVICES} and ${UCX_NET_DEVICES}"
else
    unset UCX_NET_DEVICES
    echo "Mode baseline: local rank ${local_rank} using GPU ${CUDA_VISIBLE_DEVICES}"
fi

exec "$@"
