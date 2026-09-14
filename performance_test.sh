#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scripts/lib.sh"
[[ $# == 0 ]] || die '用法：bash performance_test.sh'
check_image
check_files
service_ready
client /deploy/scripts/performance.py
