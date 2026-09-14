#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scripts/lib.sh"
[[ $# == 0 ]] || die '用法：bash gpu_kernel_test.sh'
check_image
check_files
run_id=$(date +%Y%m%d-%H%M%S)
dc run --rm --pull never --network none --gpus all --env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  --mount "type=bind,src=$PACKAGE_DIR,dst=/deploy,readonly" \
  --entrypoint /opt/r3/bin/python "$IMAGE" -c 'import torch,pytest; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability()==(8,0); raise SystemExit(pytest.main(["/deploy/tests/gpu/test_vision_routing.py","-q","-k","bias_vl or padding","-p","no:cacheprovider"]))' \
  2>&1 | tee "$RESULTS/gpu-vision-routing-$run_id.log"
