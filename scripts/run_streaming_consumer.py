"""CLI wrapper for the MQTT streaming ingest consumer.

Usage:
    python scripts/run_streaming_consumer.py --config configs/streaming.yaml --sink parquet
    python scripts/run_streaming_consumer.py --sink null --score
"""

from aether_pdm.ingest.streaming import main

if __name__ == "__main__":
    main()
