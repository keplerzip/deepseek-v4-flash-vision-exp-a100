#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scripts/lib.sh"
check_image
check_files
service_ready
run_id=$(date +%Y%m%d-%H%M%S)-$$
[[ $# -le 1 ]] || die '用法：bash acceptance.sh [standard|long|saturation]'
extra=()
case ${1:-standard} in
 standard) ;;
 long) extra=(--long) ;;
 saturation) extra=(--long --saturation) ;;
 *) die '用法：bash acceptance.sh [standard|long|saturation]' ;;
esac
client /deploy/scripts/acceptance.py --base http://host.docker.internal:8005 --assets /deploy/tests/multimodal --output "/results/acceptance-$run_id.jsonl" "${extra[@]}"
