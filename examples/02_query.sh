#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://localhost:8000}
SESSION_ID=${SESSION_ID:-demo-session}
QUESTION=${QUESTION:-"What does the sample document talk about?"}
TOP_K=${TOP_K:-3}
MAX_TOKENS=${MAX_TOKENS:-256}

curl -sS -X POST "${BASE_URL}/sessions/${SESSION_ID}/query" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"${QUESTION//\"/\\\"}\", \"top_k\": ${TOP_K}, \"max_tokens\": ${MAX_TOKENS}}"
