"""Module-level default timeout for HTTP requests."""

_default_timeout: float = 120.0  # 2 minutes
_flex_timeout: float = 600.0  # 10 minutes


def set_default_timeout(seconds: float) -> None:
    """Set the default HTTP request timeout (in seconds)."""
    global _default_timeout
    _default_timeout = seconds


def get_default_timeout() -> float:
    """Return the current default HTTP request timeout (in seconds)."""
    return _default_timeout


def set_flex_timeout(seconds: float) -> None:
    """Set the HTTP timeout for flex/async requests (in seconds)."""
    global _flex_timeout
    _flex_timeout = seconds


def get_flex_timeout() -> float:
    """Return the current flex request timeout (in seconds)."""
    return _flex_timeout
