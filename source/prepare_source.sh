#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
[[ $# == 0 ]] || { printf 'Usage: bash source/prepare_source.sh\n' >&2; exit 1; }
commit=cf7898691b58820a8ba98e018f612d4a0c2f69f0
printf '%s  %s\n' \
  6215aefdca4c187a7e900cef0e07a1f83770734d250c32a84ca71de01cdc09f0 "$root/source/r3-port-from-r2.3.patch" \
  531b7a867bf2391d25be37c49ff34172b6186bd8f4b16f295e8f9532c1633d4a "$root/source/deepseek_v4_renderer.py" | sha256sum --check --quiet
[[ ! -e "$root/source/vllm-source.tar.gz" ]] || { printf 'Existing source archive retained; choose a fresh checkout to rebuild.\n' >&2; exit 1; }
mkdir -p "$root/work"
stage=$(mktemp -d "$root/work/source-XXXXXX")
git init -q "$stage"
git -C "$stage" remote add origin https://github.com/haosdent/vllm.git
git -C "$stage" fetch --depth 1 origin "$commit"
git -C "$stage" checkout --detach FETCH_HEAD
[[ $(git -C "$stage" rev-parse HEAD) == "$commit" ]]
git -C "$stage" apply --check "$root/source/r3-port-from-r2.3.patch"
git -C "$stage" apply "$root/source/r3-port-from-r2.3.patch"
cp "$root/source/deepseek_v4_renderer.py" "$stage/vllm/renderers/deepseek_v4.py"
# Runtime Python code plus native extension source; no old build payloads.
archive=$(mktemp "$root/source/.vllm-source-XXXXXX.tar.gz")
trap 'rm -f -- "$archive"' EXIT
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='*.so' --exclude='*.whl' \
  -czf "$archive" -C "$stage" vllm csrc LICENSE
mv -- "$archive" "$root/source/vllm-source.tar.gz"
printf 'SOURCE_PREPARED=%s\nCHECKOUT=%s\n' "$root/source/vllm-source.tar.gz" "$stage"
