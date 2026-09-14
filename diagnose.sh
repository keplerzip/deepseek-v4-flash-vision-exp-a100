#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scripts/lib.sh"
[[ $# == 0 ]] || die '用法：bash diagnose.sh'
run_id=$(date +%Y%m%d-%H%M%S)
dir=$RESULTS/diagnose-$run_id
mkdir -p "$dir"
cp "$PACKAGE_DIR/config/service.json" "$dir/config.json"
dc version > "$dir/docker-version.txt" 2>&1 || true
dc network inspect bridge --format '{{json .IPAM}}' > "$dir/bridge.json" 2>&1 || true
if exists "$CONTAINER"; then
  owned "$CONTAINER"
  dc exec "$CONTAINER" /opt/r3/bin/python /deploy/scripts/effective_state.py > "$dir/effective-state.json.log" 2>&1 || true
  dc logs --timestamps --tail 20000 "$CONTAINER" > "$dir/server.log" 2>&1 || true
  dc inspect --format '{"id":"{{.Id}}","image":"{{.Image}}","state":{{json .State}},"ports":{{json .HostConfig.PortBindings}}}' "$CONTAINER" > "$dir/container.json"
fi
check_image
dc run --rm --pull never --network none --gpus all --entrypoint nvidia-smi "$IMAGE" -q > "$dir/nvidia-smi.txt" 2>&1 || true
# A narrow safe inventory avoids exporting environment/API keys.
client /deploy/scripts/health.py http://host.docker.internal:8005 > "$dir/docker-internal-health.txt" 2>&1 || true
dc run --rm --pull never --network bridge --add-host host.docker.internal:host-gateway --env R3_API_KEY= --env VLLM_API_KEY= \
  --entrypoint /opt/r3/bin/python "$IMAGE" -c 'import os,urllib.request; h={"Authorization":"Bearer "+os.environ["R3_API_KEY"]} if os.environ.get("R3_API_KEY") else {}; print(urllib.request.urlopen(urllib.request.Request("http://host.docker.internal:8005/metrics",headers=h),timeout=10).read().decode())' > "$dir/metrics.txt" 2>&1 || true
tar -czf "$RESULTS/diagnose-$run_id.tar.gz" -C "$RESULTS" "diagnose-$run_id"
log "$RESULTS/diagnose-$run_id.tar.gz"
