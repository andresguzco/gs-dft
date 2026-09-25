"""Shared helpers for the experiment runners: system loading, SplatKS assembly, the Gaussian
reference, the RESULT line, probes and plot styling.

Importing this package enables float64 before any dftax constant is built. The submodules are not
imported eagerly, so a plotter that only needs ``common.results`` or ``common.style`` does not load
dftax. Runners are invoked as modules::

    uv run python -m experiments.<experiment>.<runner> experiment=<config> key=value
"""
import jax

jax.config.update("jax_enable_x64", True)
