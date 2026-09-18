#!/usr/bin/env bash
# Optional derived image for archiving; deployment uses the tested overlay path.
set -euo pipefail
source "$(dirname -- "$0")/../scripts/lib.sh"
check_image
check_files
dc build --network none --pull=false --build-arg "BASE_IMAGE=$IMAGE" \
  --tag dsv4-flash-a100:20260916-r3.4 --file "$PACKAGE_DIR/source/Dockerfile" "$PACKAGE_DIR/source"
log 'DERIVED_IMAGE_BUILT; the deployment continues to use the pinned base image plus verified overlay'
