#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://localhost:8000}
SESSION_ID=${SESSION_ID:-demo-session}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

curl -sS -X POST "${BASE_URL}/sessions/${SESSION_ID}/ingest" \
  -F "files=@${SCRIPT_DIR}/sample_document.txt"
