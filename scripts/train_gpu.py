"""CLI wrapper for single-GPU (or CPU-fallback) torch anomaly training.

Usage:
    python scripts/train_gpu.py \
        --features data/interim/features/features_v2.parquet \
        --mlflow-uri sqlite:///mlflow.db --epochs 50 --device auto
"""

from aether_pdm.ops.gpu_train import main

if __name__ == "__main__":
    main()
