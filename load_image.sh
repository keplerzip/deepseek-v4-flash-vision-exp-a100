#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scripts/lib.sh"
check_image
log 'R3.4 使用已存在的冻结镜像和只读源码覆盖；无需导入新镜像。'
