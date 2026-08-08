#!/bin/bash
set -e

# Wait for MLflow to be reachable. Poll the /health endpoint (returns 200
# once the tracking server is up); the root path / can redirect or return
# non-200 depending on the MLflow version/proxy setup, which would make this
# loop spin forever.
echo "Waiting for MLflow at ${AETHER_MLFLOW_TRACKING_URI:-http://mlflow:5000}..."
until python -c "
import urllib.request
base = '${AETHER_MLFLOW_TRACKING_URI:-http://mlflow:5000}'.rstrip('/')
try:
    urllib.request.urlopen(base + '/health', timeout=2)
except Exception:
    exit(1)
" 2>/dev/null; do
    echo "  MLflow not ready, retrying..."
    sleep 2
done
echo "MLflow reachable."

# Initialize DB
python -c "from aether_pdm.db.database import init_db; init_db(); print('DB initialized.')"

# Start API
exec uvicorn aether_pdm.serve.app:app --host 0.0.0.0 --port 8000
