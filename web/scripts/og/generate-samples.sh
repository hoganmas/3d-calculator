#!/usr/bin/env bash
# Generate a batch of example OG images for visual review before deploying.
#
# Usage: web/scripts/og/generate-samples.sh [siteUrl] [outDir]
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
web_dir="$(cd "$script_dir/../.." && pwd)"

cd "$web_dir"
npx tsx scripts/og/generate-samples.mjs "$@"
