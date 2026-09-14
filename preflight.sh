#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scripts/lib.sh"
[[ $# == 0 ]] || die '用法：bash preflight.sh；R3.3 不再需要方案编号'
check_image
check_files
[[ $MODEL_DIR == /* && -d $MODEL_DIR ]] || die '请在 deployment.env 中设置正确的 MODEL_DIR 绝对路径'
dc run --rm --pull never --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$MODEL_DIR,dst=/model,readonly" \
  --mount "type=bind,src=$PACKAGE_DIR,dst=/deploy,readonly" \
  --entrypoint /opt/r3/bin/python "$IMAGE" /deploy/scripts/verify_model.py /model > "$RESULTS/model-headers.json"
inventory=$(dc run --rm --pull never --network none --gpus all --entrypoint nvidia-smi "$IMAGE" --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits)
mapfile -t gpus <<< "$inventory"
[[ ${#gpus[@]} == 8 ]] || die '需要 8 张 A100 80GB'
for row in "${gpus[@]}"; do
  IFS=',' read -r name mem driver <<< "$row"
  mem=${mem//[[:space:]]/}; driver=${driver//[[:space:]]/}
  [[ $name == *A100* && $mem =~ ^[0-9]+$ ]] && ((mem >= 80000)) || die "GPU 不符合要求：$row"
  [[ $(printf '%s\n' 580.126.20 "$driver" | sort -V | head -1) == 580.126.20 ]] || die "驱动低于 580.126.20：$driver"
done
printf '%s\n' "$inventory" > "$RESULTS/gpu-inventory.csv"
dc run --rm --pull never --network none --gpus all "${R33_MOUNTS[@]}" \
  --mount "type=bind,src=$PACKAGE_DIR,dst=/deploy,readonly" \
  --mount "type=bind,src=$MODEL_DIR,dst=/model,readonly" \
  --entrypoint /opt/r3/bin/python "$IMAGE" /deploy/scripts/gpu_config_check.py > "$RESULTS/gpu-config.json.log"
log 'PREFLIGHT=PASS R3.3；尚未执行模型推理'
