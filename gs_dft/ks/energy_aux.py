"""The per-term energy breakdown returned alongside the total.

Vendored here (dftax 0.7 removed its `utils.energy_aux`) so the definition is ours to hold.
The field order is load-bearing — callers unpack positionally — and the trailing optional
fields stay so every tuple built or unpacked anywhere keeps the same shape.
"""
from __future__ import annotations

from typing import NamedTuple

from jaxtyping import Scalar


class EnergyAux(NamedTuple):
    """Canonical ordering for energy auxiliary data."""

    kinetic: Scalar
    hartree: Scalar
    xc: Scalar
    external: Scalar
    nelec: Scalar
    kl_div: Scalar | None = None
    tv_dist: Scalar | None = None
    entropy: Scalar | None = None


def pack_energy_aux(
    *,
    kinetic: Scalar,
    hartree: Scalar,
    xc: Scalar,
    external: Scalar,
    nelec: Scalar,
    kl_div: Scalar | None = None,
    tv_dist: Scalar | None = None,
    entropy: Scalar | None = None,
) -> EnergyAux:
    """Return the energy aux tuple in the canonical order.

    Keyword-only on purpose: eight fields of the same type are exactly the signature where a
    positional call silently swaps two of them and the error surfaces as a wrong energy breakdown
    rather than an exception.
    """
    return EnergyAux(
        kinetic=kinetic,
        hartree=hartree,
        xc=xc,
        external=external,
        nelec=nelec,
        kl_div=kl_div,
        tv_dist=tv_dist,
        entropy=entropy,
    )
