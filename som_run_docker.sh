#!/bin/bash

# https://registry.ngc.nvidia.com/orgs/nvidian/containers/nemo-rl/tags

# export NRL_FORCE_REBUILD_VENVS=true

cd /home/smajumdar/PycharmProjects/nemo-rl-public

docker run --gpus all --rm -it \
  --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=24g \
  --privileged --pid=host \
  -v "$PWD:$PWD" \
  -v "/media/smajumdar/data/huggingface:/media/smajumdar/data/huggingface" \
  -e HF_HOME="/media/smajumdar/data/huggingface" \
  -e HF_DATASETS_CACHE="/media/smajumdar/data/huggingface/datasets" \
  -e UV_CACHE_DIR="$PWD/.cache/uv" \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e HF_TOKEN=$HF_TOKEN \
  -w "$PWD" \
  nvcr.io/nvidian/nemo-rl:7b30253-45711825 bash
