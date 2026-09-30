#!/usr/bin/env bash
# Submit the WSL ablation worker from macOS without keeping the SSH session open.

set -Eeuo pipefail

SSH_HOST="wsl"
REMOTE_PROJECT=""
REMOTE_OUTPUT=""
BASE_URL="http://localhost:8000/v1"
CONDA_ENV="medllm"
SEED="42"
ENDPOINT_WAIT_MINUTES="30"
BARK_URL_FILE=""
SYNC=1
NO_START_VLLM=0

usage() {
  echo "Usage: $0 [options]"
  echo "  --project-dir PATH           Project path inside WSL; auto-detected if omitted"
  echo "  --output-dir PATH            Output path inside WSL; timestamped default if omitted"
  echo "  --ssh-host HOST              SSH config alias (default: wsl)"
  echo "  --base-url URL               vLLM endpoint visible inside WSL"
  echo "  --conda-env NAME             Conda environment (default: medllm)"
  echo "  --seed N                     Blinded-export seed"
  echo "  --endpoint-wait-minutes N    Wait for endpoint before failing"
  echo "  --bark-url-file PATH         Optional path inside WSL containing Bark URL"
  echo "  --no-sync                    Skip the default git pull --ff-only"
  echo "  --no-start-vllm              Require an already-running matching endpoint"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir) REMOTE_PROJECT="$2"; shift 2 ;;
    --output-dir) REMOTE_OUTPUT="$2"; shift 2 ;;
    --ssh-host) SSH_HOST="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --endpoint-wait-minutes) ENDPOINT_WAIT_MINUTES="$2"; shift 2 ;;
    --bark-url-file) BARK_URL_FILE="$2"; shift 2 ;;
    --no-sync) SYNC=0; shift ;;
    --no-start-vllm) NO_START_VLLM=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v ssh >/dev/null 2>&1 || { echo "ssh is not available" >&2; exit 1; }
ssh "$SSH_HOST" "true"

if [[ -z "$REMOTE_PROJECT" ]]; then
  REMOTE_PROJECT="$(ssh "$SSH_HOST" 'for p in "$HOME/repo/med_dict" "$HOME/med_dict"; do if [ -d "$p/.git" ] && [ -f "$p/run.py" ]; then printf "%s\n" "$p"; exit 0; fi; done; while IFS= read -r p; do if [ -d "$p/.git" ] && [ -f "$p/run.py" ]; then printf "%s\n" "$p"; exit 0; fi; done < <(find "$HOME" -maxdepth 4 -type d -name med_dict 2>/dev/null)')"
fi
if [[ -z "$REMOTE_PROJECT" ]]; then
  echo "Could not auto-detect the WSL project. Pass --project-dir." >&2
  exit 1
fi

if [[ -z "$REMOTE_OUTPUT" ]]; then
  remote_stamp="$(ssh "$SSH_HOST" 'date +%Y%m%d_%H%M%S')"
  REMOTE_OUTPUT="$REMOTE_PROJECT/results/staged_ablation_annotated_$remote_stamp"
fi

quote() { printf '%q' "$1"; }
remote_command="bash -s -- $(quote "$REMOTE_PROJECT") $(quote "$REMOTE_OUTPUT") $(quote "$BASE_URL") $(quote "$CONDA_ENV") $(quote "$SEED") $(quote "$ENDPOINT_WAIT_MINUTES") $(quote "$BARK_URL_FILE") $(quote "$SYNC") $(quote "$NO_START_VLLM")"

launch_result="$(ssh "$SSH_HOST" "$remote_command" <<'REMOTE_LAUNCH'
set -Eeuo pipefail
project_dir="$1"
output_dir="$2"
base_url="$3"
conda_env="$4"
seed="$5"
wait_minutes="$6"
bark_url_file="$7"
sync_enabled="$8"
no_start_vllm="$9"

test -d "$project_dir/.git"
test -f "$project_dir/run.py"
if [[ "$sync_enabled" == "1" ]]; then
  echo "Syncing WSL checkout with git pull --ff-only" >&2
  GIT_TERMINAL_PROMPT=0 git -C "$project_dir" pull --ff-only >&2
fi
test -f "$project_dir/staged_ablation.py"
test -f "$project_dir/scripts/run_staged_ablation_wsl.sh"
mkdir -p "$output_dir"
worker_args=(
  --project-dir "$project_dir"
  --output-dir "$output_dir"
  --base-url "$base_url"
  --conda-env "$conda_env"
  --seed "$seed"
  --endpoint-wait-minutes "$wait_minutes"
)
if [[ -n "$bark_url_file" ]]; then
  worker_args+=(--bark-url-file "$bark_url_file")
fi
if [[ "$no_start_vllm" == "1" ]]; then
  worker_args+=(--no-start-vllm)
fi
nohup bash "$project_dir/scripts/run_staged_ablation_wsl.sh" "${worker_args[@]}" \
  > "$output_dir/nohup.log" 2>&1 < /dev/null &
pid=$!
printf '%s|%s|%s\n' "$pid" "$output_dir" "$output_dir/nohup.log"
REMOTE_LAUNCH
)"

IFS='|' read -r remote_pid resolved_output nohup_log <<< "$launch_result"
echo "Submitted WSL ablation worker"
echo "PID: $remote_pid"
echo "Output: $resolved_output"
echo "Log: $nohup_log"
echo "Monitor log: ssh $SSH_HOST \"tail -f '$nohup_log'\""
echo "Check status: ssh $SSH_HOST \"cat '$resolved_output/status.tsv'\""
echo "Check process: ssh $SSH_HOST \"ps -p $remote_pid -o pid,etime,cmd\""
