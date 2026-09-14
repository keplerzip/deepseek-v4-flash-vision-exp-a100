#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
source "$root/scripts/lib.sh"
check_image
check_files
work=$(mktemp -d "${TMPDIR:-/tmp}/dsv4-r33-rebuild.XXXXXX")
printf 'BUILD_DIR=%s\n' "$work"
mkdir -p "$work/src" "$work/native/output" "$work/context/vllm"
tar -xzf "$root/source/vllm-source.tar.gz" -C "$work/src"
cp "$root/source/deepseek_v4_renderer.py" "$work/src/vllm/renderers/deepseek_v4.py"
cp "$root/source/native/build.py" "$root/source/native/vision_bindings.cpp" "$work/native/"
dc run --rm --pull never --network none --env TORCH_CUDA_ARCH_LIST=8.0 --env MAX_JOBS=8 \
  --mount "type=bind,src=$work/src,dst=/src,readonly" --mount "type=bind,src=$work/native,dst=/build" \
  --entrypoint /opt/r3/bin/python "$IMAGE" /build/build.py
cp -a "$work/src/vllm/." "$work/context/vllm/"
cp "$work/native/output/_r3_vision_ops.so" "$work/context/vllm/"
cat > "$work/context/Dockerfile" <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
LABEL com.deepseek.rebuilt="operator-local;requires-revalidation"
DOCKERFILE
dc build --pull=false --network none --build-arg "BASE_IMAGE=$IMAGE" -t dsv4-flash-a100:r3.3-local-rebuild "$work/context"
printf 'LOCAL_REBUILD=dsv4-flash-a100:r3.3-local-rebuild; requires separate GPU validation and image identity approval before use with pinned deployment scripts.\n'
