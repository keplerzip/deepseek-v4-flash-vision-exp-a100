#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scripts/lib.sh"
state="$RUNTIME/control/upgrade-state.env"
[[ -f $state ]] || die '没有安装时的旧服务状态记录'
source "$state"
if [[ -n $PREVIOUS_ID ]]; then
  [[ $(dc inspect --format '{{.Id}}' "$PREVIOUS_CONTAINER") == "$PREVIOUS_ID" ]] || die '旧容器已变更；未停止当前服务'
fi
bash "$PACKAGE_DIR/stop.sh"
if [[ $PREVIOUS_RUNNING == true ]]; then
  dc start "$PREVIOUS_CONTAINER" >/dev/null
  log "ROLLBACK_STARTED=$PREVIOUS_CONTAINER; wait for its health check"
else
  log 'ROLLBACK=PASS; prior service was stopped or absent and remains so'
fi
