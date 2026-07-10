#!/usr/bin/env bash
# Deploy offline-trained TD/BCQ weights into the standard online run directory.
#
# Keeps configs/tasks/<task>/online_rl.yaml unchanged; only replaces artifacts under
# runs/<task>/ so Machine B can resume with run_start_machineB.sh.
#
# Usage:
#   bash run_deploy_offline_to_online.sh
#   OFFLINE_DIR=runs/screw_sorting/offline_train_bcq_phase-warmup bash run_deploy_offline_to_online.sh
#   RLT_ACTOR_TAG=final RLT_COPY_REPLAY=0 bash run_deploy_offline_to_online.sh
#
# After deploy:
#   bash run_stop_training.sh          # if services are running
#   bash run_start_machineB.sh
#   bash run_start_machineA.sh
#   bash run_start_rollout.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TASK="${RLT_TASK:-screw_sorting}"
TASK_DIR="${RLT_TASK_DIR:-runs/${TASK}}"
TASK_CONFIG="${RLT_TASK_CONFIG:-configs/tasks/${TASK}/online_rl.yaml}"

# Offline bundle directory (output of offline_train_from_replay.py).
OFFLINE_DIR="${RLT_OFFLINE_DIR:-${TASK_DIR}/offline_train_bcq_phase-warmup}"

# Actor artifact: best (default) | final | bundle
ACTOR_TAG="${RLT_ACTOR_TAG:-best}"

# 1 = copy offline filtered replay; 0 = keep existing runs/<task>/replay/replay_journal.pkl
COPY_REPLAY="${RLT_COPY_REPLAY:-1}"

# 1 = backup existing checkpoints/actor_snapshot/replay before overwrite
DO_BACKUP="${RLT_DO_BACKUP:-1}"

usage() {
  cat <<EOF
Usage: bash run_deploy_offline_to_online.sh [options]

Copy offline training artifacts into ${TASK_DIR} for online continuation.
Online config stays at ${TASK_CONFIG}.

Options:
  -h, --help              Show this help
  -n, --dry-run           Print actions without copying files

Environment:
  RLT_TASK / RLT_TASK_DIR
  RLT_OFFLINE_DIR         Offline train dir (default: ${TASK_DIR}/offline_train_bcq_phase-warmup)
  RLT_ACTOR_TAG           best | final | bundle (default: best)
  RLT_COPY_REPLAY         1 copy offline replay, 0 keep online replay (default: 1)
  RLT_DO_BACKUP           1 backup existing artifacts first (default: 1)

Resolved paths:
  offline:  ${OFFLINE_DIR}
  online:   ${TASK_DIR}
  config:   ${TASK_CONFIG}
EOF
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -n|--dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

_run() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

_resolve_actor_source() {
  case "${ACTOR_TAG}" in
    best)
      echo "${OFFLINE_DIR}/best_actor_snapshot.pkl"
      ;;
    final)
      echo "${OFFLINE_DIR}/final_actor_snapshot.pkl"
      ;;
    bundle)
      echo "${OFFLINE_DIR}/actor_snapshot/actor_snapshot.pkl"
      ;;
    *)
      echo "Invalid RLT_ACTOR_TAG=${ACTOR_TAG}; use best, final, or bundle." >&2
      exit 1
      ;;
  esac
}

OFFLINE_DIR="$(cd "${OFFLINE_DIR}" && pwd)"
TASK_DIR="$(mkdir -p "${TASK_DIR}" && cd "${TASK_DIR}" && pwd)"

CHECKPOINT_SRC="${OFFLINE_DIR}/checkpoints/latest.pkl"
ACTOR_SRC="$(_resolve_actor_source)"
REPLAY_SRC="${OFFLINE_DIR}/replay/replay_journal.pkl"

CHECKPOINT_DST="${TASK_DIR}/checkpoints/latest.pkl"
ACTOR_DST="${TASK_DIR}/actor_snapshot/actor_snapshot.pkl"
REPLAY_DST="${TASK_DIR}/replay/replay_journal.pkl"

if [[ ! -f "${CHECKPOINT_SRC}" ]]; then
  echo "Missing learner checkpoint: ${CHECKPOINT_SRC}" >&2
  exit 1
fi
if [[ ! -f "${ACTOR_SRC}" ]]; then
  echo "Missing actor snapshot (${ACTOR_TAG}): ${ACTOR_SRC}" >&2
  exit 1
fi
if [[ "${COPY_REPLAY}" == "1" && ! -f "${REPLAY_SRC}" ]]; then
  echo "Missing offline replay journal: ${REPLAY_SRC}" >&2
  exit 1
fi
if [[ ! -f "${TASK_CONFIG}" ]]; then
  echo "Missing task config: ${TASK_CONFIG}" >&2
  exit 1
fi

if [[ "${DO_BACKUP}" == "1" ]]; then
  BACKUP_DIR="${TASK_DIR}/backups/deploy_$(date +%Y%m%d_%H%M%S)"
  echo "==> Backup existing online artifacts to ${BACKUP_DIR}"
  _run mkdir -p "${BACKUP_DIR}"
  for item in checkpoints actor_snapshot replay; do
    if [[ -e "${TASK_DIR}/${item}" ]]; then
      _run cp -a "${TASK_DIR}/${item}" "${BACKUP_DIR}/"
    fi
  done
fi

echo "==> Deploy offline bundle -> online run dir"
echo "    offline:     ${OFFLINE_DIR}"
echo "    online:      ${TASK_DIR}"
echo "    actor tag:   ${ACTOR_TAG}"
echo "    copy replay: ${COPY_REPLAY}"

_run mkdir -p "${TASK_DIR}/checkpoints" "${TASK_DIR}/actor_snapshot" "${TASK_DIR}/replay"

echo "==> checkpoint"
echo "    ${CHECKPOINT_SRC} -> ${CHECKPOINT_DST}"
_run cp -f "${CHECKPOINT_SRC}" "${CHECKPOINT_DST}"

echo "==> actor snapshot"
echo "    ${ACTOR_SRC} -> ${ACTOR_DST}"
_run cp -f "${ACTOR_SRC}" "${ACTOR_DST}"

if [[ "${COPY_REPLAY}" == "1" ]]; then
  echo "==> replay journal"
  echo "    ${REPLAY_SRC} -> ${REPLAY_DST}"
  _run cp -f "${REPLAY_SRC}" "${REPLAY_DST}"
else
  echo "==> replay journal unchanged (${REPLAY_DST})"
fi

cat <<EOF

Deploy complete.

Online config (unchanged):
  ${TASK_CONFIG}

Next steps:
  1. Stop running services if any:
       bash run_stop_training.sh
  2. Start Machine B with task config:
       bash run_start_machineB.sh
  3. Start Machine A + rollout as usual.

Verify after Machine B starts:
  - learner log restores checkpoint step=20000 (or your offline steps)
  - replay restores transitions from ${REPLAY_DST}
  - learner status shows ready_for_online=true once replay size >= warmup_min_size
EOF
