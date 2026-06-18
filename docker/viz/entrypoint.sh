#!/bin/sh
set -eu

OUTPUT_DIR="${OUTPUT_DIR:-/app/data/pipeline}"
STATIC_DIR="${STATIC_DIR:-/app/static/timeline_viz}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

if [ ! -f "${OUTPUT_DIR}/aggregated_complete.csv" ]; then
  echo "ERROR: Pipeline output not found at ${OUTPUT_DIR}" >&2
  echo "Expected aggregated_complete.csv and related artifacts." >&2
  echo "Mount pipeline data at OUTPUT_DIR or bake it into the image at build time." >&2
  exit 1
fi

export OUTPUT_DIR STATIC_DIR PYTHONPATH="/app:/app/src"

echo "Timeline viz listening on http://${HOST}:${PORT}"
echo "Pipeline output: ${OUTPUT_DIR}"

exec uvicorn broadcast_pipeline.viz.asgi:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --log-level info
