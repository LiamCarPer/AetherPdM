"""CLI wrapper for batch scoring.

Usage:
    python scripts/run_batch_scorer.py [--org acme] [--hysteresis 3] [--cooldown-min 30]
"""

from aether_pdm.ops.batch_scorer import main

if __name__ == "__main__":
    main()
