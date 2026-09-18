#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scripts/lib.sh"
[[ $# == 0 ]] || die 'Usage: bash newapi_test.sh'
: "${R34_NEWAPI_URL:?Set R34_NEWAPI_URL to your New API root URL}"
: "${R34_NEWAPI_TOKEN_FILE:?Set R34_NEWAPI_TOKEN_FILE to a file with a model token}"
token_file=$(realpath -e -- "$R34_NEWAPI_TOKEN_FILE")
[[ -f $token_file && -r $token_file ]] || die 'New API token file not readable'
export R34_NEWAPI_URL
check_image
check_files
dc run --rm --pull never --network host --user "$(id -u):$(id -g)" \
  --env "R34_NEWAPI_URL=$R34_NEWAPI_URL" --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$token_file,dst=/newapi-token,readonly" \
  --mount "type=bind,src=$PACKAGE_DIR,dst=/deploy,readonly" \
  --mount "type=bind,src=$RESULTS,dst=/results" \
  --entrypoint /opt/r3/bin/python "$IMAGE" /deploy/scripts/gateway_probe.py
