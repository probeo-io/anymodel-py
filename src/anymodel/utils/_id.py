"""ID generation utilities."""

import secrets


def generate_id(prefix: str = "gen") -> str:
    """Generate a random ID with the given prefix."""
    return f"{prefix}-{secrets.token_urlsafe(12)}"
