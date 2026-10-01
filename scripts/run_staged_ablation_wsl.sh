#!/usr/bin/env bash
# Run all four extraction-ablation variants on WSL. This script is intended to
# be launched by submit_staged_ablation_wsl.sh and is safe for a long nohup run.

set -Eeuo pipefail

PROJECT_DIR=""
OUTPUT_DIR=""
BASE_URL="http://localhost:8000/v1"
CONDA_ENV="medllm"
SEED="42"
ENDPOINT_WAIT_MINUTES="30"
BARK_URL_FILE=""
POLL_SECONDS="60"
PYTHON_BIN="python3"
NO_START_VLLM=0
VLLM_STARTED=0
VLLM_PID=""
VLLM_LOG=""

usage() {
  echo "Usage: $0 [--project-dir PATH] --output-dir PATH [options]"
  echo "  --base-url URL               vLLM OpenAI-compatible endpoint"
  echo "  --conda-env NAME             Conda environment (default: medllm)"
  echo "  --seed N                     Blinded-export randomization seed"
  echo "  --endpoint-wait-minutes N    Wait for vLLM, polling every 60s"
  echo "  --no-start-vllm              Never start a server; only wait for an existing one"
  echo "  --bark-url-file PATH         Optional remote file containing Bark base URL"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --endpoint-wait-minutes) ENDPOINT_WAIT_MINUTES="$2"; shift 2 ;;
    --no-start-vllm) NO_START_VLLM=1; shift ;;
    --bark-url-file) BARK_URL_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "$PROJECT_DIR" ]]; then
  PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "--output-dir is required" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR/logs"
MAIN_LOG="$OUTPUT_DIR/orchestrator.log"
STATUS_FILE="$OUTPUT_DIR/status.tsv"
PID_FILE="$OUTPUT_DIR/worker.pid"
ORCHESTRATION_MANIFEST="$OUTPUT_DIR/orchestration_manifest.tsv"
exec > >(tee -a "$MAIN_LOG") 2>&1

timestamp() { date --iso-8601=seconds; }
status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >> "$STATUS_FILE"; }

BARK_URL="${BARK_URL:-}"
if [[ -n "$BARK_URL_FILE" && -r "$BARK_URL_FILE" ]]; then
  BARK_URL="$(tr -d '\r\n' < "$BARK_URL_FILE")"
fi

notify() {
  local message="$1"
  if [[ -z "$BARK_URL" ]]; then
    return 0
  fi
  local encoded
  encoded="$($PYTHON_BIN -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$message")"
  curl -fsS "${BARK_URL%/}/Codex/${encoded}" >/dev/null || true
}

CURRENT_TASK="preflight"
SUCCESS=0
on_exit() {
  local rc=$?
  if [[ $VLLM_STARTED -eq 1 && -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(timestamp)] Stopping vLLM server started by this worker (pid=$VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "$VLLM_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$VLLM_PID" 2>/dev/null; then
      kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
  fi
  if [[ $SUCCESS -eq 1 ]]; then
    return 0
  fi
  status "FAILED" "$CURRENT_TASK rc=$rc"
  notify "Ablation failed during $CURRENT_TASK (rc=$rc). See $MAIN_LOG"
  exit "$rc"
}
trap on_exit EXIT

echo "$$" > "$PID_FILE"
status "STARTED" "worker_pid=$$"
notify "Ablation worker submitted. Preflight and GPU queue started. Output: $OUTPUT_DIR"

echo "[$(timestamp)] Preflight"
git -C "$PROJECT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || { echo "Not a git project: $PROJECT_DIR"; exit 1; }
[[ -f "$PROJECT_DIR/staged_ablation.py" ]] || { echo "Missing staged_ablation.py in $PROJECT_DIR"; exit 1; }
[[ -f "$PROJECT_DIR/scripts/run_staged_ablation_wsl.sh" ]] || { echo "Missing worker script"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is not available"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is not available"; exit 1; }
command -v flock >/dev/null 2>&1 || { echo "flock is not available"; exit 1; }

if ! command -v conda >/dev/null 2>&1; then
  for conda_sh in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -r "$conda_sh" ]]; then
      # shellcheck disable=SC1090
      source "$conda_sh"
      break
    fi
  done
fi
command -v conda >/dev/null 2>&1 || { echo "conda is not available"; exit 1; }
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" "$PROJECT_DIR/staged_ablation.py" --help >/dev/null
EXPECTED_MODEL="$($PYTHON_BIN -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["model"]["name"])' "$PROJECT_DIR/exp/v31_breast_annotated_test.yaml")"

printf 'created_at\t%s\nproject_dir\t%s\noutput_dir\t%s\nbase_url\t%s\nconda_env\t%s\nmodel\t%s\nseed\t%s\n' \
  "$(timestamp)" "$PROJECT_DIR" "$OUTPUT_DIR" "$BASE_URL" "$CONDA_ENV" "$EXPECTED_MODEL" "$SEED" \
  > "$ORCHESTRATION_MANIFEST"

for cancer in breast pdac; do
  for variant in A B C D; do
    if [[ -e "$OUTPUT_DIR/$cancer/$variant/manifest.json" || -e "$OUTPUT_DIR/$cancer/$variant/progress.json" ]]; then
      echo "Existing run detected at $OUTPUT_DIR/$cancer/$variant; refusing to overwrite" >&2
      exit 1
    fi
  done
done

df -h "$OUTPUT_DIR"
available_kb="$(df -Pk "$OUTPUT_DIR" | awk 'NR==2 {print $4}')"
if [[ -z "$available_kb" || "$available_kb" -lt 1048576 ]]; then
  echo "Less than 1 GiB is available under $OUTPUT_DIR" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

# Serialize med_dict GPU work launched through this script. External jobs that do
# not use this lock are handled by the nvidia-smi queue below.
exec 9>/tmp/med_dict_ablation_gpu.lock
while ! flock -n 9; do
  echo "[$(timestamp)] Another med_dict GPU worker holds the lock; waiting 60s"
  sleep "$POLL_SECONDS"
done

endpoint_ready() {
  curl -fsS "${BASE_URL%/}/models" | "$PYTHON_BIN" -c '
import json, sys
expected = sys.argv[1]
payload = json.load(sys.stdin)
ids = [str(item.get("id", "")) for item in payload.get("data", [])]
raise SystemExit(0 if expected in ids else 1)
' "$EXPECTED_MODEL" >/dev/null
}

gpu_busy() {
  [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')" ]]
}

wait_for_endpoint() {
  local label="$1"
  local wait_seconds=$((ENDPOINT_WAIT_MINUTES * 60))
  local elapsed=0
  while ! endpoint_ready; do
    if (( elapsed >= wait_seconds )); then
      echo "Expected model $EXPECTED_MODEL unavailable after ${ENDPOINT_WAIT_MINUTES} minutes at $BASE_URL ($label)" >&2
      return 1
    fi
    if [[ $VLLM_STARTED -eq 1 && -n "$VLLM_PID" ]] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vLLM server exited before becoming ready. Tail of $VLLM_LOG:" >&2
      tail -n 100 "$VLLM_LOG" >&2 || true
      return 1
    fi
    echo "[$(timestamp)] Waiting for vLLM endpoint: $BASE_URL ($label)"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
    sleep "$POLL_SECONDS"
    elapsed=$((elapsed + POLL_SECONDS))
  done
}

if ! endpoint_ready; then
  if [[ $NO_START_VLLM -eq 1 ]]; then
    wait_for_endpoint "--no-start-vllm"
  else
    endpoint_host="$($PYTHON_BIN -c 'import sys,urllib.parse; print(urllib.parse.urlparse(sys.argv[1]).hostname or "")' "$BASE_URL")"
    endpoint_port="$($PYTHON_BIN -c 'import sys,urllib.parse; print(urllib.parse.urlparse(sys.argv[1]).port or 8000)' "$BASE_URL")"
    if [[ "$endpoint_host" != "localhost" && "$endpoint_host" != "127.0.0.1" && "$endpoint_host" != "0.0.0.0" ]]; then
      echo "Endpoint is remote ($endpoint_host); this worker will not start a server there"
      wait_for_endpoint "remote endpoint"
    else
      while gpu_busy; do
        if endpoint_ready; then
          break
        fi
        echo "[$(timestamp)] GPU is occupied and no matching endpoint is ready; queued for 60s"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
        sleep "$POLL_SECONDS"
      done

      if ! endpoint_ready; then
        command -v vllm >/dev/null 2>&1 || { echo "vllm command is unavailable in $CONDA_ENV"; exit 1; }
        VLLM_LOG="$OUTPUT_DIR/vllm_server.log"
        echo "[$(timestamp)] GPU is free; starting $EXPECTED_MODEL on port $endpoint_port"
        vllm serve "$EXPECTED_MODEL" \
          --host 0.0.0.0 \
          --port "$endpoint_port" \
          --enable-prefix-caching \
          --max-model-len 16384 \
          --gpu-memory-utilization 0.85 \
          --trust-remote-code \
          --dtype float16 > "$VLLM_LOG" 2>&1 &
        VLLM_PID=$!
        VLLM_STARTED=1
        echo "$VLLM_PID" > "$OUTPUT_DIR/vllm_server.pid"
        status "VLLM_STARTING" "pid=$VLLM_PID log=$VLLM_LOG"
        wait_for_endpoint "server startup"
        status "VLLM_READY" "pid=$VLLM_PID"
      fi
    fi
  fi
fi
echo "[$(timestamp)] vLLM endpoint ready: $BASE_URL"
status "PREFLIGHT_OK" "project=$PROJECT_DIR endpoint=$BASE_URL env=$CONDA_ENV"
notify "Ablation started. Expected runtime about 2-4 hours. Output: $OUTPUT_DIR"

run_one() {
  local variant="$1"
  local cancer="$2"
  local task_log="$OUTPUT_DIR/logs/${variant}_${cancer}.log"
  CURRENT_TASK="variant=$variant cancer=$cancer"
  status "RUNNING" "$CURRENT_TASK"
  echo "[$(timestamp)] Starting $CURRENT_TASK"

  "$PYTHON_BIN" "$PROJECT_DIR/staged_ablation.py" generate \
    --cancer "$cancer" \
    --sample-set annotated \
    --variant "$variant" \
    --output-dir "$OUTPUT_DIR/$cancer" \
    --base-url "$BASE_URL" > "$task_log" 2>&1 &
  local child_pid=$!
  echo "$child_pid" > "$OUTPUT_DIR/logs/${variant}_${cancer}.pid"

  while kill -0 "$child_pid" 2>/dev/null; do
    local completed="0"
    if [[ -f "$OUTPUT_DIR/$cancer/$variant/outputs.jsonl" ]]; then
      completed="$(wc -l < "$OUTPUT_DIR/$cancer/$variant/outputs.jsonl" | tr -d ' ')"
    elif [[ -f "$OUTPUT_DIR/$cancer/$variant/progress.json" ]]; then
      completed="$($PYTHON_BIN -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("completed_indices", [])))' "$OUTPUT_DIR/$cancer/$variant/progress.json" 2>/dev/null || echo 0)"
    fi
    echo "[$(timestamp)] $CURRENT_TASK pid=$child_pid completed=$completed/20"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
    sleep "$POLL_SECONDS"
  done

  if ! wait "$child_pid"; then
    echo "[$(timestamp)] Failed: $CURRENT_TASK. Tail of $task_log:" >&2
    tail -n 80 "$task_log" >&2 || true
    return 1
  fi
  status "COMPLETED" "$CURRENT_TASK"
  echo "[$(timestamp)] Completed $CURRENT_TASK"
}

# Variant is the outer loop so the study executes in the intended A→B→C→D order.
for variant in A B C D; do
  for cancer in breast pdac; do
    run_one "$variant" "$cancer"
  done
done

CURRENT_TASK="blinded judge export"
for cancer in breast pdac; do
  "$PYTHON_BIN" "$PROJECT_DIR/staged_ablation.py" export-judge \
    --study-dir "$OUTPUT_DIR/$cancer" \
    --output-dir "$OUTPUT_DIR/$cancer/judge" \
    --seed "$SEED"
done

status "COMPLETED" "all variants and blinded exports"
notify "Ablation generation completed. Output: $OUTPUT_DIR"
SUCCESS=1
echo "[$(timestamp)] All staged ablation runs completed"
