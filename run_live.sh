#!/bin/bash
# Launch the full Blender pipeline in GUI mode so camera motion is visible in
# the viewport.
#
# Usage:
#   bash run_live.sh              # Full pipeline
#   bash run_live.sh --skip-vlm   # Baseline capture only
#
# API keys are read from the environment. For example:
#   export ANTHROPIC_API_KEY="..."
#   export DASHSCOPE_API_KEY="..."

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER="${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
SCRIPT="$PROJECT_ROOT/scripts/run_pipeline.py"
CONFIG="$PROJECT_ROOT/pipeline_config.json"

"$BLENDER" \
  --python "$SCRIPT" \
  -- \
  --config "$CONFIG" \
  "$@"
