#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scripts/lib.sh"
check_image
check_files
service_ready
run_id=$(date +%Y%m%d-%H%M%S)-$$
mode=${1:-full}
[[ $# -le 1 && ( $mode == smoke || $mode == full ) ]] || die '用法：bash multimodal_test.sh [smoke|full]'
client /deploy/scripts/multimodal_suite.py --base http://host.docker.internal:8005 --assets /deploy/tests/multimodal --mode "$mode" --repeat "${R3_MM_REPEAT:-1}" --output "/results/multimodal-$mode-$run_id.jsonl"
