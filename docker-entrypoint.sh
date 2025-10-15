#!/bin/sh
set -euo pipefail

# Ensure the Chroma persistence directory ownership matches the running user.
if [ -d "/chroma_db" ]; then
    TARGET_UID="$(id -u)"
    TARGET_GID="$(id -g)"
    CURRENT_OWNER="$(stat -c '%u:%g' /chroma_db || echo '')"
    DESIRED_OWNER="${TARGET_UID}:${TARGET_GID}"
    if [ "${CURRENT_OWNER}" != "${DESIRED_OWNER}" ]; then
        chown -R "${DESIRED_OWNER}" /chroma_db
    fi
fi

exec "$@"
