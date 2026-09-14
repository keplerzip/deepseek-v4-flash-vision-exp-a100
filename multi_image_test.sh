#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scripts/lib.sh"
[[ $# == 0 ]] || die 'Usage: bash multi_image_test.sh'
check_image
check_files
service_ready
client /deploy/scripts/multi_image_probe.py
