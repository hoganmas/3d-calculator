#!/usr/bin/env bash
# Render an example OG PNG for a laplaci share URL, using the same headless
# render pipeline api/og.ts uses in production.
#
# Usage: web/scripts/og/render-from-url.sh <url> [outPath]
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
web_dir="$(cd "$script_dir/../.." && pwd)"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <url> [outPath]" >&2
  exit 1
fi

cd "$web_dir"
npx tsx scripts/og/render-from-url.mjs "$@"
