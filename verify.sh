#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scripts/lib.sh"
[[ $# == 0 ]] || die '用法：bash verify.sh'
exec 9>"$RUNTIME/control/start-stop.lock"
flock -n 9 || die '另一个启停或验收正在进行'
check_image
check_files
service_ready
run_id=$(date +%Y%m%d-%H%M%S)-$$
report="$RESULTS/verify-$run_id"
mkdir -p "$report"
exec > >(tee "$report/run.log") 2>&1
failed=0
run_check(){
  local name=$1 rc=0
  shift
  log "CHECK=$name"
  "$@" || rc=$?
  printf '%s\t%s\n' "$name" "$rc" >> "$report/status.tsv"
  if ((rc)); then failed=1; log "CHECK_FAIL=$name RC=$rc"; fi
}
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
initial_id=$(dc inspect --format '{{.Id}}' "$CONTAINER")
run_check effective-state dc exec "$CONTAINER" /opt/r3/bin/python /deploy/scripts/effective_state.py
run_check full-multimodal bash "$PACKAGE_DIR/multimodal_test.sh" full
run_check protocol-tools-stream bash "$PACKAGE_DIR/protocol_test.sh"
run_check responses-input-errors client /deploy/scripts/responses_error_probe.py
run_check tool-history-tokenization client /deploy/scripts/r39_protocol_probe.py
run_check codex-image-tool-history bash "$PACKAGE_DIR/codex_image_test.sh"
run_check five-and-eight-images bash "$PACKAGE_DIR/multi_image_test.sh"
run_check cache-three-protocols bash "$PACKAGE_DIR/cache_test.sh"
run_check ten-rounds-c32 bash "$PACKAGE_DIR/concurrency_test.sh"
run_check protocols-concurrency-long bash "$PACKAGE_DIR/acceptance.sh" long
# All TP ranks must have validated the loaded quantized stacked projection.
# Read startup logs separately; the subsequent log scan covers this verification.
dc logs "$CONTAINER" > "$report/startup.log" 2>&1
for rank in {0..7}; do
  if ! grep -Eq "R34_STACKED_WKV_CHECK=PASS tp_rank=$rank([[:space:]]|$)" "$report/startup.log"; then
    log "R39_WKV_RANK_CHECK=MISSING RANK=$rank"; failed=1
  fi
  if ! grep -Eq "R35_DSPARK_BROADCAST_READY tp_rank=$rank([[:space:]]|$)" "$report/startup.log"; then
    log "R39_BROADCAST_RANK_CHECK=MISSING RANK=$rank"; failed=1
  fi
done
run_check health client /deploy/scripts/health.py http://host.docker.internal:8005
run_check final-effective-state dc exec "$CONTAINER" /opt/r3/bin/python /deploy/scripts/effective_state.py
dc logs --timestamps --since "$started" "$CONTAINER" > "$report/server.log" 2>&1 || failed=1
if grep -nE 'RuntimeError: Already borrowed|CUDA out of memory|torch.OutOfMemoryError|Engine core initialization failed|The matcher has terminated after accepting the stop token' "$report/server.log"; then failed=1; fi
[[ $(dc inspect --format '{{.Id}}' "$CONTAINER") == "$initial_id" ]] || failed=1
[[ $(dc inspect --format '{{.State.Status}} {{.RestartCount}}' "$CONTAINER") == 'running 0' ]] || failed=1
dc exec "$CONTAINER" nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv > "$report/gpu.csv" || failed=1
cat "$report/gpu.csv"
if ((failed)); then
  log "R3.9_VERIFY=FAIL REPORT=$report"
  log '失败结果已保留；本验收不修改配置、不自动降为 0.88、不重试改判、不切换方案。'
  exit 1
fi
log "R3.9_VERIFY=PASS MODEL=DeepSeek-V4-Flash MEMORY=0.92 CONTEXT=1048576 CONCURRENCY=32 MULTIMODAL=ON IMAGE_LIMIT=999 REPORT=$report"
log 'FULL_WINDOW_C32=NOT_TESTED NEWAPI_BILLING_END_TO_END=NOT_TESTED'
