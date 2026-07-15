#!/usr/bin/env bash
# Ergonomic headless Squiggle runner.
#   tools/squiggle/run.sh path/to/model.squiggle
# Wraps the squiggle-lang CLI (v0.10.0), which is installed locally in
# tools/squiggle/node_modules. Works around the published package's broken
# bin shebang by invoking the CLI entrypoint directly with node.
#
# First-time / after a clean checkout: run `npm install` in this directory
# (package.json pins @quri/squiggle-lang + date-fns, an undeclared dep of the
# CLI that must be present or it errors with ERR_MODULE_NOT_FOUND).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="$HERE/node_modules/@quri/squiggle-lang/dist/cli/index.js"
if [[ ! -f "$CLI" ]]; then
  echo "squiggle CLI not found; run: npm --prefix '$HERE' install" >&2
  exit 1
fi
exec node "$CLI" run "$@"
