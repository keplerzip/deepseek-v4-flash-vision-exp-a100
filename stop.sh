#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scripts/lib.sh"
[[ $# == 0 ]] || die '用法：bash stop.sh'
exec 9>"$RUNTIME/control/start-stop.lock"
flock -n 9 || die '另一个 R3.3 启停正在进行'
if exists "$CONTAINER"; then owned "$CONTAINER"; dc stop --time 90 "$CONTAINER"; fi
