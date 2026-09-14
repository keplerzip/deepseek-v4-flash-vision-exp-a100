#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
[[ $# == 0 ]] || { echo '用法：bash deploy.sh'; exit 1; }
bash "$root/load_image.sh"
bash "$root/gpu_kernel_test.sh"
bash "$root/start.sh"
bash "$root/verify.sh"
