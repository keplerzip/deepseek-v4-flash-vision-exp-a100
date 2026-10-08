#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scripts/lib.sh"
check_image
service_ready
client /deploy/scripts/codex_image_probe.py
