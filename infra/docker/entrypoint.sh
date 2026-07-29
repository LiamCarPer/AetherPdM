#!/bin/bash
set -e

# Wait for MLflow to be reachable
echo "Waiting for MLflow at ${AETHER_MLFLOW_TRACKING_URI:-http://mlflow:5000}..."
until python -c "
import urllib.request
uri = '${AETHER_MLFLOW_TRACKING_URI:-http://mlflow:5000}'
try:
    urllib.request.urlopen(uri, timeout=2)
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
