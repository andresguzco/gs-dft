"""Suite-wide setup: float64 and no GPU preallocation, before any test module imports jax.

``slow`` marks the reference and finite-difference tests, so ``-m "not slow"`` is a fast loop.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
jax.config.update("jax_enable_x64", True)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
