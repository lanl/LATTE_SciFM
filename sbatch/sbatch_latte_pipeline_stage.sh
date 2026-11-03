#!/bin/bash
set -euo pipefail
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
PYTHON=${LATTE_PYTHON:?Set the installed LATTE Python executable}
if [[ "$1" == prepare ]]; then
  exec "$PYTHON" scripts/latte_pipeline.py "$@"
fi
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
exec "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" scripts/latte_pipeline_gpu.py "$@"
