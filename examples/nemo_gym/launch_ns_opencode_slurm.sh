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
Launch NeMo RL + NeMo Gym on Slurm using ray.sub with a NeMo RL runtime container.

Required environment variables:
  SLURM_ACCOUNT         Slurm account for sbatch
  SLURM_PARTITION       Slurm partition for sbatch

Optional environment variables:
  REPO_LOCATION         NeMo RL repo root (default: git root of current dir)
  EXP_NAME              Slurm job name (default: ns-opencode-nemo-gym)
  NUM_ACTOR_NODES       Number of Ray actor nodes (default: 1)
  NUM_SLURM_NODES       Number of Slurm nodes to request (default: NUM_ACTOR_NODES)
  GPUS_PER_NODE         GPUs per node (default: 8)
  SLURM_TIME            Time limit (default: 4:0:0)
  CONTAINER_IMAGE       Container image for NeMo RL cluster runtime
                        (default: nvcr.io/nvidian/nemo-rl:7b30253-45711825)
  OPENCODE_SERVER_CONTAINER
                        Docker image used to run ns-opencode server workloads
                        (default: gitlab-master.nvidia.com/smajumdar/nemo_containers/ns-opencode:2026.02.18)
  MOUNTS                Explicit ray.sub mounts string (SRC:DST[,SRC:DST...])
  EXTRA_MOUNTS          Additional mounts appended to default mounts

  RUN_MODE              interactive|batch (default: interactive)
  RUN_COMMAND           Command run on head node when RUN_MODE=batch
  NEMO_GYM_RUN_CONFIG   Config path for default batch command
                        (default: examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml)
  EXTRA_OVERRIDES       Extra CLI overrides appended to default batch command

Examples:
  # 1) Start an interactive Ray+container cluster (no training command yet)
  SLURM_ACCOUNT=myacct \
  SLURM_PARTITION=compute \
  NUM_ACTOR_NODES=2 \
  bash examples/nemo_gym/launch_ns_opencode_slurm.sh

  # 2) Launch one-shot NeMo Gym training command in the same container
  SLURM_ACCOUNT=myacct \
  SLURM_PARTITION=compute \
  RUN_MODE=batch \
  NUM_ACTOR_NODES=2 \
  EXTRA_OVERRIDES="logger.wandb_enabled=True" \
  bash examples/nemo_gym/launch_ns_opencode_slurm.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_LOCATION="${REPO_LOCATION:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
RAY_SUB_PATH="${REPO_LOCATION}/ray.sub"

if [[ ! -f "${RAY_SUB_PATH}" ]]; then
  echo "Error: could not find ray.sub at ${RAY_SUB_PATH}" >&2
  exit 1
fi

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT before launching}"
: "${SLURM_PARTITION:?Set SLURM_PARTITION before launching}"

EXP_NAME="${EXP_NAME:-ns-opencode-nemo-gym}"
NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-1}"
FINAL_NUM_SLURM_NODES="${NUM_SLURM_NODES:-${NUM_ACTOR_NODES}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
SLURM_TIME="${SLURM_TIME:-4:0:0}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidian/nemo-rl:7b30253-45711825}"
OPENCODE_SERVER_CONTAINER="${OPENCODE_SERVER_CONTAINER:-gitlab-master.nvidia.com/smajumdar/nemo_containers/ns-opencode:2026.02.18}"
RUN_MODE="${RUN_MODE:-interactive}"
RUN_COMMAND="${RUN_COMMAND:-}"
NEMO_GYM_RUN_CONFIG="${NEMO_GYM_RUN_CONFIG:-examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

if [[ "${RUN_MODE}" != "interactive" && "${RUN_MODE}" != "batch" ]]; then
  echo "Error: RUN_MODE must be one of: interactive, batch" >&2
  exit 1
fi

mount_root="$(findmnt -n -o TARGET --target "${REPO_LOCATION}" || true)"
mount_root="${mount_root:-${REPO_LOCATION}}"
default_mounts="${mount_root}:${mount_root}"

if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
  default_mounts+="${default_mounts:+,}${EXTRA_MOUNTS}"
fi

MOUNTS="${MOUNTS:-${default_mounts}}"

if [[ "${RUN_MODE}" == "batch" && -z "${RUN_COMMAND}" ]]; then
  read -r -d '' RUN_COMMAND <<EOF_BATCH || true
cd ${REPO_LOCATION}
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config ${NEMO_GYM_RUN_CONFIG} \
  ++cluster.num_nodes=${NUM_ACTOR_NODES} \
  ++cluster.gpus_per_node=${GPUS_PER_NODE} \
  ${EXTRA_OVERRIDES}
EOF_BATCH
fi

if [[ "${RUN_MODE}" == "interactive" ]]; then
  COMMAND=""
else
  COMMAND="${RUN_COMMAND}"
fi

cd "${REPO_LOCATION}"

echo "Submitting Slurm job with the following settings:"
echo "  REPO_LOCATION=${REPO_LOCATION}"
echo "  EXP_NAME=${EXP_NAME}"
echo "  NUM_ACTOR_NODES=${NUM_ACTOR_NODES}"
echo "  FINAL_NUM_SLURM_NODES=${FINAL_NUM_SLURM_NODES}"
echo "  GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "  SLURM_TIME=${SLURM_TIME}"
echo "  CONTAINER_IMAGE=${CONTAINER_IMAGE}"
echo "  OPENCODE_SERVER_CONTAINER=${OPENCODE_SERVER_CONTAINER}"
echo "  RUN_MODE=${RUN_MODE}"
echo "  MOUNTS=${MOUNTS}"
if [[ -n "${COMMAND}" ]]; then
  echo "  COMMAND=${COMMAND}"
else
  echo "  COMMAND=<empty> (interactive ray cluster)"
fi

COMMAND="${COMMAND}" \
CONTAINER="${CONTAINER_IMAGE}" \
MOUNTS="${MOUNTS}" \
GPUS_PER_NODE="${GPUS_PER_NODE}" \
sbatch \
  --nodes="${FINAL_NUM_SLURM_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --partition="${SLURM_PARTITION}" \
  --time="${SLURM_TIME}" \
  --job-name="${EXP_NAME}" \
  --gres="gpu:${GPUS_PER_NODE}" \
  "${RAY_SUB_PATH}"
