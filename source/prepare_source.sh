#!/usr/bin/env bash
# Export the installed frozen runtime's Python sources, then apply the overlay.
# This is an image-based source reconstruction, not a fresh upstream build.
set -euo pipefail
source "$(dirname -- "$0")/../scripts/lib.sh"
check_image
out="$PACKAGE_DIR/source/vllm-source"
[[ ! -e $out ]] || die 'source/vllm-source 已存在；保留现有内容，不覆盖'
mkdir -p "$out"
dc run --rm --pull never --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$out,dst=/export" --entrypoint /opt/r3/bin/python "$IMAGE" -c '
from pathlib import Path
import shutil
root=Path("/usr/local/lib/python3.12/dist-packages/vllm")
for source in root.rglob("*.py"):
    destination=Path("/export/vllm")/source.relative_to(root)
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,destination)
'
cp -a "$PACKAGE_DIR/source/overlay/." "$out/"
log "SOURCE_EXPORTED=$out (Python runtime source; existing compiled dependencies remain in base image)"
