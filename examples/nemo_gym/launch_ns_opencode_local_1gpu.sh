#!/usr/bin/env bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

usage() {
  cat <<'USAGE'
Launch NeMo Gym locally on 1 GPU using the running NeMo RL Docker container.

Behavior:
- If run inside a container, executes command directly.
- If run on host, docker-execs into a running nemo-rl container.

Optional environment variables:
  REPO_LOCATION             Repo root in host/container (default: git root or cwd)
  NEMO_RL_CONTAINER         Container ID/name to use (default: auto-detect by image)
  CONTAINER_IMAGE           Image used for auto-detect
                            (default: nvcr.io/nvidian/nemo-rl:7b30253-45711825)
  LOCAL_CUDA_VISIBLE_DEVICES
                            CUDA_VISIBLE_DEVICES for local execution (default: 0)
  LOCAL_GPUS_PER_NODE       Value for ++cluster.gpus_per_node (default: 1)
  LOCAL_NUM_NODES           Value for ++cluster.num_nodes (default: 1)
  NEMO_GYM_RUN_CONFIG       Config path for default command
                            (default: examples/nemo_gym/grpo_qwen3_1p7b_base.yaml)
  TRAIN_DATA_PATH           Train dataset path override for default command
                            (default: examples/nemo_gym/data/workplace_assistant_example_with_agent_ref.jsonl)
  VAL_DATA_PATH             Validation dataset path override for default command
                            (default: examples/nemo_gym/data/workplace_assistant_example_with_agent_ref.jsonl)
  RUN_COMMAND               If set, run this command instead of default
  EXTRA_OVERRIDES           Extra overrides appended to default command

Examples:
  # Run default NeMo Gym command on 1 GPU in running container
  bash examples/nemo_gym/launch_ns_opencode_local_1gpu.sh

  # Run custom command on the same container
  RUN_COMMAND='uv run python -c "print(123)"' \
  bash examples/nemo_gym/launch_ns_opencode_local_1gpu.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_LOCATION="${REPO_LOCATION:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidian/nemo-rl:7b30253-45711825}"
NEMO_RL_CONTAINER="${NEMO_RL_CONTAINER:-}"
LOCAL_CUDA_VISIBLE_DEVICES="${LOCAL_CUDA_VISIBLE_DEVICES:-0}"
LOCAL_GPUS_PER_NODE="${LOCAL_GPUS_PER_NODE:-1}"
LOCAL_NUM_NODES="${LOCAL_NUM_NODES:-1}"
NEMO_GYM_RUN_CONFIG="${NEMO_GYM_RUN_CONFIG:-examples/nemo_gym/grpo_qwen3_1p7b_base.yaml}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-examples/nemo_gym/data/workplace_assistant_example_with_agent_ref.jsonl}"
VAL_DATA_PATH="${VAL_DATA_PATH:-examples/nemo_gym/data/workplace_assistant_example_with_agent_ref.jsonl}"
RUN_COMMAND="${RUN_COMMAND:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

if [[ -z "${RUN_COMMAND}" ]]; then
  read -r -d '' RUN_COMMAND <<EOF_CMD || true
cd ${REPO_LOCATION}
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config ${NEMO_GYM_RUN_CONFIG} \
  ++cluster.num_nodes=${LOCAL_NUM_NODES} \
  ++cluster.gpus_per_node=${LOCAL_GPUS_PER_NODE} \
  data.train.data_path=${TRAIN_DATA_PATH} \
  data.validation.data_path=${VAL_DATA_PATH} \
  logger.wandb_enabled=False \
  ${EXTRA_OVERRIDES}
EOF_CMD
fi

echo "Local launch settings:"
echo "  REPO_LOCATION=${REPO_LOCATION}"
echo "  CONTAINER_IMAGE=${CONTAINER_IMAGE}"
echo "  LOCAL_CUDA_VISIBLE_DEVICES=${LOCAL_CUDA_VISIBLE_DEVICES}"
echo "  LOCAL_GPUS_PER_NODE=${LOCAL_GPUS_PER_NODE}"
echo "  LOCAL_NUM_NODES=${LOCAL_NUM_NODES}"
echo "  TRAIN_DATA_PATH=${TRAIN_DATA_PATH}"
echo "  VAL_DATA_PATH=${VAL_DATA_PATH}"
echo "  RUN_COMMAND=${RUN_COMMAND}"

if [[ -f /.dockerenv ]]; then
  echo "Detected container environment; running command directly."
  CUDA_VISIBLE_DEVICES="${LOCAL_CUDA_VISIBLE_DEVICES}" bash -lc "${RUN_COMMAND}"
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is required when running from host." >&2
  exit 1
fi

if [[ -z "${NEMO_RL_CONTAINER}" ]]; then
  NEMO_RL_CONTAINER="$(docker ps --filter "ancestor=${CONTAINER_IMAGE}" --format '{{.ID}}' | head -n1)"
fi

if [[ -z "${NEMO_RL_CONTAINER}" ]]; then
  echo "Error: no running container found for image ${CONTAINER_IMAGE}." >&2
  echo "Set NEMO_RL_CONTAINER=<container-id-or-name> explicitly." >&2
  exit 1
fi

echo "Using container: ${NEMO_RL_CONTAINER}"
docker exec \
  -e CUDA_VISIBLE_DEVICES="${LOCAL_CUDA_VISIBLE_DEVICES}" \
  "${NEMO_RL_CONTAINER}" \
  bash -lc "${RUN_COMMAND}"
