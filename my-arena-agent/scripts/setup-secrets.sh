#!/usr/bin/env bash
set -euo pipefail

# Initialize secrets submodule and copy .env into project root (if present)
# Usage: ./scripts/setup-secrets.sh

SECRETS_SUBMODULE_DIR="secrets"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -d "$ROOT_DIR/$SECRETS_SUBMODULE_DIR" ]; then
  echo "secrets submodule already present"
else
  echo "Initializing secrets submodule (you must have access to the private repo)"
  git submodule update --init --remote || true
fi

if [ -f "$ROOT_DIR/$SECRETS_SUBMODULE_DIR/.env" ]; then
  cp "$ROOT_DIR/$SECRETS_SUBMODULE_DIR/.env" "$ROOT_DIR/.env"
  chmod 600 "$ROOT_DIR/.env"
  echo "Copied .env from secrets submodule"
else
  echo "No .env found in secrets submodule; ensure a private repo with .env exists at 'secrets/.env'"
fi
