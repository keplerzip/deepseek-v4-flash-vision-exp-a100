#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scripts/lib.sh"
[[ $# == 0 ]] || die 'Usage: bash load_image.sh'
check_files
(cd "$PACKAGE_DIR" && sha256sum --check images/SHA256SUMS)
dc load --input "$PACKAGE_DIR/images/dsv4-flash-a100-r3.3.docker.tar"
check_image
log 'R3.3 完整镜像导入及校验完成'
