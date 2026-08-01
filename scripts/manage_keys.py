"""CLI wrapper for API key management.

Usage:
    python scripts/manage_keys.py create --name demo
    python scripts/manage_keys.py list
    python scripts/manage_keys.py revoke --id 1
"""

from aether_pdm.ops.apikeys import main

if __name__ == "__main__":
    main()
