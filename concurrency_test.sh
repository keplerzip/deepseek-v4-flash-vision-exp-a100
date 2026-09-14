#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scripts/lib.sh"
[[ $# == 0 ]] || die '此脚本不接受参数'
check_image
check_files
service_ready
client /deploy/scripts/concurrency_probe.py
