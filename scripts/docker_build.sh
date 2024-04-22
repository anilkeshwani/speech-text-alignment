#!/usr/bin/env bash

set -euo pipefail

if [ ! -f Dockerfile ]; then
    echo "Error: Dockerfile not found in the current directory. Run this script from the project root."
    exit 1
fi

DOCKER_IMAGE_NAME='sta'
VERSION=$(git describe --tags --dirty --always)

docker build \
    --progress=plain \
    -t "${DOCKER_IMAGE_NAME}:${VERSION}" \
    -t "${DOCKER_IMAGE_NAME}:latest" \
    .
