#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$PACKAGE_DIR/deployment.env"
IMAGE=$(cat "$PACKAGE_DIR/runtime/base-image-id" 2>/dev/null || true)
OWNER=deepseek-v4-flash-r3.9-20260929
CONTAINER=dsv4-flash-r39
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
  [[ $(dc inspect --format '{{index .Config.Labels "com.deepseek.owner"}}' "$1") == "$OWNER" ]] || die "同名容器不属于本 R3.9 包：$1"
  [[ $(dc inspect --format '{{range .Mounts}}{{if eq .Destination "/deploy"}}{{.Source}}{{end}}{{end}}' "$1") == "$PACKAGE_DIR" ]] || die '该容器由另一目录管理；请使用原目录的脚本'
}
R39_MOUNTS=()
check_image(){
  check_files
  [[ $IMAGE =~ ^sha256:[0-9a-f]{64}$ ]] || die '尚未选定旧服务镜像；请先运行 bash install.sh'
  dc image inspect "$IMAGE" >/dev/null 2>&1 || die '本机缺少安装时选定的旧服务镜像'
  local image_id relative expected_release
  image_id=$(dc image inspect --format '{{.Id}}' "$IMAGE")
  grep -F -x -- "$image_id" "$PACKAGE_DIR/images/accepted-image-ids.txt" >/dev/null || die '冻结基础镜像 ID 不在清单'
  expected_release=$(awk -v id="$image_id" '$1 == id {print $2}' "$PACKAGE_DIR/images/runtime-images.tsv")
  [[ -n $expected_release && $(dc image inspect --format '{{index .Config.Labels "com.deepseek.release"}}' "$IMAGE") == "$expected_release" ]] || die '冻结基础镜像版本标签不匹配'
  R39_MOUNTS=()
  while IFS= read -r relative; do
    [[ $relative == vllm/* && $relative != *..* && $relative != *,* ]] || die '非法 overlay 路径'
    R39_MOUNTS+=(--mount "type=bind,src=$PACKAGE_DIR/source/overlay/$relative,dst=/usr/local/lib/python3.12/dist-packages/$relative,readonly")
  done < "$PACKAGE_DIR/source/overlay-files.txt"
}
check_files(){ (cd "$PACKAGE_DIR" && sha256sum --check --quiet FILES.sha256); }
client(){
  dc run --rm --pull never "${R39_MOUNTS[@]}" --network "${R3_CLIENT_NETWORK:-bridge}" \
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
