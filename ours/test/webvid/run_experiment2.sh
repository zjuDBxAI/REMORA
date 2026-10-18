#!/usr/bin/env bash
# Persistent experiment-2 launcher; safe to rerun unchanged to resume completed stages.
set -uo pipefail
remora_repo="${REMORA_REPO:-/home/xzl/REMORA}"
cd "$remora_repo" || exit 1
ablation_output="${ABLATION_OUTPUT:-outputs/webvid/remora-ablation-b160-v1}"
mkdir -p "$ablation_output" || exit 1
export GPU="${GPU:-2}"
bash ours/scripts/run_webvid_ablation.sh 2>&1 | tee -a "$ablation_output/run.log"
runner_rc="${PIPESTATUS[0]}"
printf '\n[runner exit] %s\n' "$runner_rc"
exit "$runner_rc"
