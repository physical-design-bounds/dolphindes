"""Shared pytest configuration.

JAX is an optional dependency (the ``jax`` extra), so the JAX-only test module is
dropped from collection entirely when JAX is not installed. Doing the skip here
rather than with ``pytest.importorskip`` inside the module lets that module use
ordinary top-level imports.

64-bit mode must be enabled before any JAX array is created, so it is set once
here, at collection time, rather than per test module.
"""

collect_ignore = []

try:
    import jax
except ImportError:  # pragma: no cover - exercised only without the jax extra
    collect_ignore.append("test_jax_fdfd.py")
else:
    jax.config.update("jax_enable_x64", True)
