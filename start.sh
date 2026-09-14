#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scripts/lib.sh"
[[ $# == 0 ]] || die '用法：bash start.sh；R3.3 只有最终部署配置'
exec 9>"$RUNTIME/control/start-stop.lock"
flock -n 9 || die '另一个 R3.3 启停正在进行'
check_image
check_files
# Reject old services and occupied ports before touching the current service.
for old in dsv4-flash-r32 dsv4-flash-r31 dsv4-vision-r3-scheme1 dsv4-vision-r3-scheme2 dsv4-vision-r3-scheme3 dsv4-vision-r3-scheme4; do
  if running "$old"; then die "旧容器 $old 仍在运行；先使用旧安装包的 stop.sh 停止，再运行本脚本；原 API key 请保留在 deployment.env"; fi
done
if exists "$CONTAINER"; then
  owned "$CONTAINER"
  if running "$CONTAINER"; then
    dc exec "$CONTAINER" /opt/r3/bin/python /deploy/scripts/effective_state.py
    client /deploy/scripts/health.py http://host.docker.internal:8005
    log 'R3.3 已运行；需要重新启动时执行 bash stop.sh 后再 bash start.sh'
    exit 0
  fi
fi
if command -v ss >/dev/null; then
  [[ -z $(ss -H -ltn 'sport = :8005') ]] || die '8005 已占用；请先停止占用服务'
fi
bash "$PACKAGE_DIR/preflight.sh"
compute=$(dc run --rm --pull never --network none --gpus all --entrypoint nvidia-smi "$IMAGE" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
[[ -z $compute ]] || die "GPU 仍有计算任务：$compute"
gateway=$(dc network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')
[[ $gateway =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || die '无法确定 Docker bridge 地址'
run_id=$(date +%Y%m%d-%H%M%S)-$$
if exists "$CONTAINER"; then
  dc logs --timestamps "$CONTAINER" > "$RESULTS/previous-$run_id.log" 2>&1 || true
  dc rm "$CONTAINER" >/dev/null
fi
dc run -d --restart unless-stopped --pull never --name "$CONTAINER" \
  --label "com.deepseek.owner=$OWNER" --label com.deepseek.role=inference \
  --network bridge --publish '127.0.0.1:8005:8005' --publish "$gateway:8005:8005" \
  --gpus all --shm-size 64g --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  "${R33_MOUNTS[@]}" \
  --mount "type=bind,src=$MODEL_DIR,dst=/model,readonly" \
  --mount "type=bind,src=$PACKAGE_DIR,dst=/deploy,readonly" \
  --mount "type=bind,src=$RUNTIME/cache,dst=/runtime-cache" \
  --env R3_API_KEY= --env VLLM_API_KEY= --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID --env VLLM_USE_V2_MODEL_RUNNER=1 \
  --env VLLM_USE_BREAKABLE_CUDAGRAPH=1 --env VLLM_SPARSE_DENSE_QUERY_BLOCK=8 \
  --env TOKENIZERS_PARALLELISM=false --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env VLLM_NO_USAGE_STATS=1 --env DO_NOT_TRACK=1 --env NCCL_DEBUG=WARN \
  --env HF_HOME=/runtime-cache/huggingface --env TRITON_CACHE_DIR=/runtime-cache/triton \
  --env TORCHINDUCTOR_CACHE_DIR=/runtime-cache/inductor --env XDG_CACHE_HOME=/runtime-cache/xdg \
  --entrypoint /opt/r3/bin/python "$IMAGE" /deploy/scripts/engine.py > "$RESULTS/container-$run_id.txt"
failed(){
  dc logs --timestamps "$CONTAINER" > "$RESULTS/startup-failed-$run_id.log" 2>&1 || true
  dc stop --time 90 "$CONTAINER" >/dev/null 2>&1 || true
  die "R3.3 启动未通过；失败容器已停止，保留日志 $RESULTS/startup-failed-$run_id.log"
}
for ((i=0;i<720;i++)); do
  state=$(dc inspect --format '{{.State.Status}} {{.RestartCount}}' "$CONTAINER")
  [[ $state == 'running 0' ]] || failed
  if dc exec "$CONTAINER" /opt/r3/bin/python /deploy/scripts/health.py >/dev/null 2>&1; then
    dc exec "$CONTAINER" /opt/r3/bin/python /deploy/scripts/effective_state.py || failed
    client /deploy/scripts/health.py http://host.docker.internal:8005 || failed
    dc logs --timestamps "$CONTAINER" > "$RESULTS/startup-$run_id.log" 2>&1
    log 'R3.3 START=PASS DeepSeek-V4-Flash；执行 bash verify.sh 完成完整验收'
    exit 0
  fi
  ((i%6)) || log "等待模型加载/编译：$((i*10)) 秒"
  sleep 10
done
failed
