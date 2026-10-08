#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT=$(cd -- "$(dirname -- "$0")" && pwd -P)
new_container=dsv4-flash-r39
new_owner=deepseek-v4-flash-r3.9-20260929
old_container=dsv4-flash-r38
old_owner=deepseek-v4-flash-r3.8-20260923
old_id=''
old_running=false
switched=0
completed=0
report=''
phase_pid=''

log(){ printf '%s\n' "$*"; }
fail(){ log "ERROR: $*" >&2; exit 1; }
for command in docker sha256sum flock timeout realpath tee; do
  command -v "$command" >/dev/null || fail "Missing command: $command"
done
if docker info >/dev/null 2>&1; then DC=(docker)
elif sudo -n docker info >/dev/null 2>&1; then DC=(sudo -n docker)
else fail 'Docker access unavailable'; fi
dc(){ "${DC[@]}" "$@"; }
exists(){ dc container inspect "$1" >/dev/null 2>&1; }
running(){ [[ $(dc inspect --format '{{.State.Running}}' "$1" 2>/dev/null) == true ]]; }

mkdir -p "$ROOT/runtime/control" "$ROOT/runtime/results"
exec 9>"$ROOT/runtime/control/install.lock"
flock -n 9 || fail 'Another R3.9 install is running'

if exists "$new_container"; then
  [[ $(dc inspect --format '{{index .Config.Labels "com.deepseek.owner"}}' "$new_container") == "$new_owner" ]] || fail 'R3.9 container name belongs to another service'
  [[ $(dc inspect --format '{{range .Mounts}}{{if eq .Destination "/deploy"}}{{.Source}}{{end}}{{end}}' "$new_container") == "$ROOT" ]] || fail 'R3.9 container belongs to another directory'
  if running "$new_container"; then
    dc exec "$new_container" /opt/r3/bin/python /deploy/scripts/effective_state.py
    log 'R39_RECHECK_RUNNING_CANDIDATE; verifying before reporting success'
    timeout --signal=TERM --kill-after=30s 7800 bash "$ROOT/verify.sh"
    log "R39_INSTALL=ALREADY_CURRENT INSTALL_DIR=$ROOT"
    exit 0
  fi
fi

if exists "$old_container"; then
  [[ $(dc inspect --format '{{index .Config.Labels "com.deepseek.owner"}}' "$old_container") == "$old_owner" ]] || fail 'Prior R3.8 container owner mismatch'
  old_id=$(dc inspect --format '{{.Id}}' "$old_container")
  old_running=$(dc inspect --format '{{.State.Running}}' "$old_container")
  old_deploy=$(dc inspect --format '{{range .Mounts}}{{if eq .Destination "/deploy"}}{{.Source}}{{end}}{{end}}' "$old_container")
  model_dir=$(dc inspect --format '{{range .Mounts}}{{if eq .Destination "/model"}}{{.Source}}{{end}}{{end}}' "$old_container")
  cache_dir=$(dc inspect --format '{{range .Mounts}}{{if eq .Destination "/runtime-cache"}}{{.Source}}{{end}}{{end}}' "$old_container")
  [[ -d $old_deploy && -d $model_dir && -d $cache_dir ]] || fail 'R3.8 deployment/model/cache mount missing'
  exec 8>"$old_deploy/runtime/control/start-stop.lock"
  flock -n 8 || fail 'R3.8 start, stop or verification is in progress'
else
  fail 'R3.8 容器不存在；请保留旧容器后再执行增量升级'
fi
[[ $model_dir == /* && -d $model_dir && $cache_dir == /* ]] || fail 'Model/cache paths must be absolute existing directories'
if [[ $cache_dir == "$ROOT/runtime/cache" ]]; then
  mkdir -p "$cache_dir"
else
  [[ -d $cache_dir ]] || fail 'Prior cache mount missing'
  if [[ -e $ROOT/runtime/cache || -L $ROOT/runtime/cache ]]; then
    [[ $(realpath -e -- "$ROOT/runtime/cache") == $(realpath -e -- "$cache_dir") ]] || fail 'R3.9 cache path already points elsewhere'
  else
    ln -s -- "$cache_dir" "$ROOT/runtime/cache"
  fi
fi
printf 'MODEL_DIR=%q\nR3_RUNTIME_DIR=\n' "$model_dir" > "$ROOT/deployment.env"
chmod 600 "$ROOT/deployment.env"

stamp=$(date +%Y%m%d-%H%M%S)-$$
report="$ROOT/runtime/control/r39-install-$stamp"
mkdir -p "$report"
exec > >(tee "$report/run.log") 2>&1
log "R39_INSTALL_ENTRY=FLAT INSTALL_DIR=$ROOT PREVIOUS=$old_container PREVIOUS_RUNNING=$old_running"

restore_previous(){
  local rc=0
  log 'R39_RESTORE: stopping candidate and restoring the previous state'
  if exists "$new_container"; then
    [[ $(dc inspect --format '{{index .Config.Labels "com.deepseek.owner"}}' "$new_container") == "$new_owner" ]] || return 1
    [[ $(dc inspect --format '{{range .Mounts}}{{if eq .Destination "/deploy"}}{{.Source}}{{end}}{{end}}' "$new_container") == "$ROOT" ]] || return 1
    dc logs --timestamps --tail 350 "$new_container" > "$report/candidate-failure.log" 2>&1 || true
    if running "$new_container"; then
      dc stop --time 90 "$new_container" >/dev/null 2>&1 || rc=1
    fi
    running "$new_container" && return 1
  fi
  if [[ $old_running != true ]]; then
    log 'R39_RESTORE=PASS PREVIOUS_RUNNING=false'
    return "$rc"
  fi
  [[ $(dc inspect --format '{{.Id}}' "$old_container") == "$old_id" ]] || return 1
  if ! running "$old_container"; then dc start "$old_container" >/dev/null || return 1; fi
  for ((i=0;i<120;i++)); do
    running "$old_container" || return 1
    if dc exec "$old_container" /opt/r3/bin/python -c 'import json,urllib.request; d=json.load(urllib.request.urlopen("http://127.0.0.1:8005/v1/models",timeout=30))["data"]; assert len(d)==1 and d[0]["id"]=="DeepSeek-V4-Flash" and d[0]["max_model_len"]==1048576' >/dev/null 2>&1; then
      log 'R39_RESTORE=PASS PREVIOUS_RUNNING=true'
      return "$rc"
    fi
    sleep 10
  done
  return 1
}
finish(){
  local rc=$? restore_rc=0
  trap - EXIT INT TERM
  set +e
  if [[ -n $phase_pid ]]; then kill -TERM "$phase_pid" 2>/dev/null; wait "$phase_pid" 2>/dev/null; fi
  if (( ! completed )); then
    (( rc )) || rc=1
    if (( switched )); then restore_previous || restore_rc=$?
    else log 'PREVIOUS_SERVICE_UNCHANGED=YES'; fi
    log "R39_INSTALL=FAIL RC=$rc RESTORE_RC=$restore_rc REPORT=$report"
  fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

log 'CHECKING_PACKAGE_FILES'
(cd "$ROOT" && sha256sum --check --quiet FILES.sha256)
image=$(dc inspect --format '{{.Image}}' "$old_container")
[[ $image =~ ^sha256:[0-9a-f]{64}$ ]] || fail 'Invalid previous image ID'
dc image inspect "$image" >/dev/null 2>&1 || fail 'The actual R3.8 image is missing; previous service unchanged'
expected_release=$(awk -v id="$image" '$1 == id {print $2}' "$ROOT/images/runtime-images.tsv")
[[ -n $expected_release && $(dc image inspect --format '{{index .Config.Labels "com.deepseek.release"}}' "$image") == "$expected_release" ]] || fail 'Unrecognized runtime image ID/release'
printf '%s\n' "$image" > "$ROOT/runtime/base-image-id"
log "R39_REUSE_IMAGE=$image BASE_RELEASE=$expected_release"

source "$ROOT/scripts/lib.sh"
check_image
log 'CHECKING_MODEL_HEADERS_AND_CPU_REGRESSIONS'
dc run --rm --pull never --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$ROOT,dst=/deploy,readonly" --mount "type=bind,src=$model_dir,dst=/model,readonly" \
  --entrypoint /opt/r3/bin/python "$image" /deploy/scripts/verify_model.py /model > "$report/model-headers.json"
dc run --rm --pull never --network none --env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 --env VLLM_TARGET_DEVICE=cpu --env HF_HUB_OFFLINE=1 --env VLLM_LOGGING_LEVEL=ERROR --env OMP_NUM_THREADS=2 \
  "${R39_MOUNTS[@]}" --user "$(id -u):$(id -g)" --mount "type=bind,src=$ROOT,dst=/deploy,readonly" --mount "type=bind,src=$model_dir,dst=/model,readonly" \
  --entrypoint /opt/r3/bin/python "$image" -m pytest -q -p no:cacheprovider \
  /deploy/tests/test_r34_runtime.py /deploy/tests/test_r35_runtime.py /deploy/tests/test_r36_xgrammar.py \
  /deploy/tests/test_r37_runtime.py /deploy/tests/test_r39_runtime.py /deploy/tests/test_r39_multimodal.py /deploy/tests/upstream /deploy/tests/test_multimodal_harness.py
dc run --rm --pull never --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$ROOT,dst=/deploy,readonly" --mount "type=bind,src=$report,dst=/report" \
  --env R38_HARNESS_REPORT=/report/multimodal-harness.json --entrypoint /opt/r3/bin/python "$image" \
  /deploy/tests/test_multimodal_harness.py
bash "$ROOT/preflight.sh"

printf 'PREVIOUS_CONTAINER=%q\nPREVIOUS_ID=%q\nPREVIOUS_RUNNING=%q\n' \
  "$old_container" "$old_id" "$old_running" > "$ROOT/runtime/control/upgrade-state.env"
chmod 600 "$ROOT/runtime/control/upgrade-state.env"
if [[ $old_running == true ]]; then
  dc exec "$old_container" /opt/r3/bin/python -c 'import json,urllib.request; d=json.load(urllib.request.urlopen("http://127.0.0.1:8005/v1/models",timeout=30))["data"]; assert len(d)==1 and d[0]["id"]=="DeepSeek-V4-Flash" and d[0]["max_model_len"]==1048576; print("PREVIOUS_HEALTH=PASS")'
  log 'R39_PERFORMANCE_BASELINE=SKIPPED_REPAIR; use performance_test.sh after upgrade'
fi

[[ $(dc inspect --format '{{.Id}}' "$old_container") == "$old_id" ]] || fail 'Previous container changed during validation; no switch performed'
[[ $(dc inspect --format '{{.State.Running}}' "$old_container") == "$old_running" ]] || fail 'Previous service state changed during validation; no switch performed'
switched=1
if [[ $old_running == true ]]; then
  log "STOPPING_PREVIOUS=$old_container"
  dc stop --time 90 "$old_container" >/dev/null
fi
run_phase(){
  local rc=0 limit=$1; shift
  timeout --signal=TERM --kill-after=30s "$limit" "$@" &
  phase_pid=$!
  wait "$phase_pid" || rc=$?
  phase_pid=''
  return "$rc"
}
log 'R39_GPU_KERNELS=UNCHANGED_FROM_R38; optional gpu_kernel_test.sh retained'
log 'STARTING_R39'
run_phase 7800 bash "$ROOT/start.sh"
log 'VERIFYING_R39_1M_C32_MULTIMODAL_PROTOCOLS'
run_phase 7800 bash "$ROOT/verify.sh"

completed=1
log "R39_INSTALL=PASS MODEL=DeepSeek-V4-Flash CONTEXT=1048576 CONCURRENCY=32 MEMORY=0.92 IMAGE_LIMIT=999 DEFAULT=DSpark-k6-V2-Graph INSTALL_DIR=$ROOT REPORT=$report"
log 'FULL_WINDOW_C32=NOT_TESTED NEWAPI_BILLING_END_TO_END=NOT_TESTED'
