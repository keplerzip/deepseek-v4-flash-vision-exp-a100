#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$PACKAGE_DIR/deployment.env"
IMAGE=dsv4-flash-a100:20260914-r3.3
OWNER=deepseek-v4-flash-r3.3-20260914
CONTAINER=dsv4-flash-r33
RUNTIME=${R3_RUNTIME_DIR:-$PACKAGE_DIR/runtime}
RESULTS=$RUNTIME/results
unset R3_API_KEY VLLM_API_KEY
log(){ printf '%s\n' "$*"; }
die(){ log "ERROR: $*" >&2; exit 1; }
if docker info >/dev/null 2>&1; then DC=(docker)
elif sudo -n docker info >/dev/null 2>&1; then DC=(sudo -n docker)
else die '无法使用 Docker；请确认 Docker / NVIDIA Container Toolkit 与当前用户权限'; fi
dc(){ "${DC[@]}" "$@"; }
umask 077
mkdir -p "$RESULTS" "$RUNTIME/cache" "$RUNTIME/control"
exists(){ dc container inspect "$1" >/dev/null 2>&1; }
running(){ [[ $(dc inspect --format '{{.State.Running}}' "$1" 2>/dev/null) == true ]]; }
owned(){
  [[ $(dc inspect --format '{{index .Config.Labels "com.deepseek.owner"}}' "$1") == "$OWNER" ]] || die "同名容器不属于本 R3.3 包：$1"
  [[ $(dc inspect --format '{{range .Mounts}}{{if eq .Destination "/deploy"}}{{.Source}}{{end}}{{end}}' "$1") == "$PACKAGE_DIR" ]] || die '该容器由另一目录管理；请使用原目录的脚本'
}
R33_MOUNTS=()
check_image(){
  dc image inspect "$IMAGE" >/dev/null 2>&1 || die '先运行 bash load_image.sh'
  local image_id
  image_id=$(dc image inspect --format '{{.Id}}' "$IMAGE")
  grep -F -x -- "$image_id" "$PACKAGE_DIR/images/accepted-image-ids.txt" >/dev/null || die 'R3.3 镜像内容 ID 不匹配'
  [[ $(dc image inspect --format '{{index .Config.Labels "com.deepseek.release"}}' "$IMAGE") == 2026.09.14-r3.3 ]] || die 'R3.3 镜像版本不匹配'
  [[ $(sha256sum "$PACKAGE_DIR/source/deepseek_v4_renderer.py" | cut -d ' ' -f 1) == 531b7a867bf2391d25be37c49ff34172b6186bd8f4b16f295e8f9532c1633d4a ]] || die 'R3.3 renderer hash mismatch'
  R33_MOUNTS=()
}
check_files(){ (cd "$PACKAGE_DIR" && sha256sum --check --quiet FILES.sha256); }
client(){
  dc run --rm --pull never --network "${R3_CLIENT_NETWORK:-bridge}" \
    --add-host host.docker.internal:host-gateway --user "$(id -u):$(id -g)" \
    --env R3_API_KEY= --env VLLM_API_KEY= \
    --mount "type=bind,src=$PACKAGE_DIR,dst=/deploy,readonly" \
    --mount "type=bind,src=$RESULTS,dst=/results" \
    --entrypoint /opt/r3/bin/python "$IMAGE" "$@"
}
service_ready(){
  owned "$CONTAINER"
  running "$CONTAINER" || die '先运行 bash start.sh'
  client /deploy/scripts/health.py http://host.docker.internal:8005
}
