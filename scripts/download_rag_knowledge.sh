#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="${GARDEN_RAG_ASSET_URL:?Set GARDEN_RAG_ASSET_URL to the GitHub Release asset URL}"
EXPECTED="e15deb0d85645943d1b74d8e92d10f1df51b63247c46858b30757d47da7ec127"
TMP_DIR="$(mktemp -d)"
ARCHIVE="$TMP_DIR/garden-rag-knowledge-v1.tar.gz"
trap 'rm -rf "$TMP_DIR"' EXIT

curl --fail --location "$URL" --output "$ARCHIVE"
printf '%s  %s\n' "$EXPECTED" "$ARCHIVE" | sha256sum --check --status
tar -xzf "$ARCHIVE" -C "$ROOT/rag"
echo "RAG knowledge installed under $ROOT/rag"
