#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
[[ $# == 0 ]] || { echo '用法：bash deploy.sh'; exit 1; }
exec bash "$root/install.sh"
