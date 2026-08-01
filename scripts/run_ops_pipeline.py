"""CLI wrapper for the scheduled ops pipeline.

Usage:
    python scripts/run_ops_pipeline.py \
        --features data/interim/features/features_v1.parquet --org acme
"""

from aether_pdm.ops.scheduler import main

if __name__ == "__main__":
    main()
