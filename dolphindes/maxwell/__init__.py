"""Public interface for TM_FDFD."""

from .maxwell_fdfd import TM_FDFD
from .maxwell_polar_fdfd import (
    TM_Polar_FDFD,
)

__all__ = [
    "TM_FDFD",
    "TM_Polar_FDFD",
]
