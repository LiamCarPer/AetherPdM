"""
API key management operations.

Usage (library):
    from aether_pdm.ops.apikeys import create_key, list_keys, revoke_key
    key = create_key(db, name="demo")
    print(key["api_key"])   # shown once

Usage (CLI):
    python -m aether_pdm.ops.apikeys create --name demo
    python -m aether_pdm.ops.apikeys list
    python -m aether_pdm.ops.apikeys revoke --id 1
"""

import argparse

from sqlalchemy.orm import Session

from aether_pdm.db.database import get_session, init_db
from aether_pdm.db.repository import create_api_key, list_api_keys, revoke_api_key
from aether_pdm.serve.auth import generate_api_key, hash_api_key


def create_key(db: Session, name: str, org: str = "default") -> dict:
    """
    Create a new API key.

    Returns dict: {id, name, org, api_key (shown ONCE), key_prefix, created_at}
    """
    full_key, prefix, secret_part = generate_api_key()
    record = create_api_key(
        db,
        name=name,
        key_prefix=prefix,
        key_hash=hash_api_key(secret_part),
        org=org,
    )
    db.commit()  # the plaintext is never persisted — only prefix + hash
    return {
        "id": record.id,
        "name": record.name,
        "org": record.org,
        "api_key": full_key,
        "key_prefix": record.key_prefix,
        "created_at": record.created_at,
    }


def list_keys(db: Session, include_revoked: bool = False) -> list[dict]:
    """Return key records as dicts (NEVER the secret/hash)."""
    records = list_api_keys(db, include_revoked=include_revoked)
    return [
        {
            "id": record.id,
            "name": record.name,
            "org": record.org,
            "key_prefix": record.key_prefix,
            "created_at": record.created_at,
            "revoked_at": record.revoked_at,
            "revoked": record.revoked_at is not None,
        }
        for record in records
    ]


def revoke_key(db: Session, key_id: int) -> dict:
    """Revoke a key by id. Raises ValueError if not found."""
    record = revoke_api_key(db, key_id)
    if record is None:
        raise ValueError(f"API key with id={key_id} not found")
    db.commit()
    return {
        "id": record.id,
        "name": record.name,
        "org": record.org,
        "key_prefix": record.key_prefix,
        "created_at": record.created_at,
        "revoked_at": record.revoked_at,
    }


def main() -> None:
    """CLI: create | list | revoke. Uses sqlite default DB."""
    parser = argparse.ArgumentParser(
        prog="apikeys",
        description="Manage AetherPdM API keys",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_p = subparsers.add_parser("create", help="Create a new API key")
    create_p.add_argument("--name", required=True, help="Human label, e.g. demo")
    create_p.add_argument("--org", default="default", help="Tenant/org seed")

    list_p = subparsers.add_parser("list", help="List API keys")
    list_p.add_argument(
        "--include-revoked",
        action="store_true",
        help="Include revoked keys",
    )

    revoke_p = subparsers.add_parser("revoke", help="Revoke an API key")
    revoke_p.add_argument("--id", type=int, required=True, help="Key record id")

    args = parser.parse_args()

    init_db()

    if args.command == "create":
        with get_session() as session:
            key = create_key(session, name=args.name, org=args.org)
        print(f"Created key '{key['name']}' (id={key['id']})")
        print(f"API key (shown once): {key['api_key']}")
        return

    if args.command == "list":
        with get_session() as session:
            keys = list_keys(session, include_revoked=args.include_revoked)
        if not keys:
            print("No API keys found.")
            return
        for k in keys:
            status_label = "revoked" if k["revoked"] else "active"
            print(
                f"id={k['id']} name={k['name']} org={k['org']} "
                f"prefix={k['key_prefix']} created_at={k['created_at']} status={status_label}"
            )
        return

    # command == "revoke"
    try:
        with get_session() as session:
            key = revoke_key(session, args.id)
    except ValueError as e:
        parser.exit(1, f"error: {e}\n")
    print(f"Revoked key id={key['id']} name={key['name']}")


if __name__ == "__main__":
    main()
