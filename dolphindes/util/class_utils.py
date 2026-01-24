"""Utility functions for class attribute validation."""

from typing import Any


def check_attributes(self: Any, *attrs: str) -> None:
    """Check that specified attributes are not None.

    Parameters
    ----------
    self : Any
        The object instance to check attributes on.
    *attrs : str
        Variable number of attribute names to check.

    Raises
    ------
    AttributeError
        If any of the specified attributes are None.
    """
    missing = [attr for attr in attrs if getattr(self, attr) is None]
    if missing:
        raise AttributeError(f"{', '.join(missing)} undefined.")
