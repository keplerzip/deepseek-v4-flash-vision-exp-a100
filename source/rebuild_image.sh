#!/usr/bin/env bash
# Optional developer rebuild; normal installation reuses the existing image.
set -euo pipefail
source "$(dirname -- "$0")/../scripts/lib.sh"
check_image
base=$IMAGE
dc image inspect "$base" >/dev/null 2>&1 || die "Optional rebuild requires the frozen base image: $base"
derived=dsv4-flash-a100:20260929-r3.9-1m-c32
dc build --network none --pull=false --provenance=false --build-arg "BASE_IMAGE=$base" \
  --tag "$derived" --file "$PACKAGE_DIR/source/Dockerfile" "$PACKAGE_DIR/source"
log "OPTIONAL_DERIVED_IMAGE_BUILT=$derived; the incremental service continues to use $base plus verified overlays"
