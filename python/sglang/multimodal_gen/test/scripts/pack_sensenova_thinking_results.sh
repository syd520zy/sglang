#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 RESULT_DIR [ARCHIVE.tar.gz]" >&2
  exit 2
fi

RESULT_DIR="$(cd "$1" && pwd)"
RESULT_PARENT="$(dirname "${RESULT_DIR}")"
RESULT_NAME="$(basename "${RESULT_DIR}")"
ARCHIVE="${2:-${RESULT_DIR}.tar.gz}"

tar -C "${RESULT_PARENT}" -czf "${ARCHIVE}" "${RESULT_NAME}"
sha256sum "${ARCHIVE}" >"${ARCHIVE}.sha256"

echo "Archive: ${ARCHIVE}"
echo "Checksum: ${ARCHIVE}.sha256"
