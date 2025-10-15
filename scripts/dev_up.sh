#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILES=(-f docker-compose.dev.yml)

if [[ "${USE_GPU:-false}" == "true" ]]; then
  COMPOSE_FILES+=(-f docker-compose.gpu.local.yml)
fi

docker compose "${COMPOSE_FILES[@]}" up -d --build "$@"
