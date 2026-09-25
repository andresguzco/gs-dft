"""The two-electron engines: streaming exact Coulomb (custom-vjp) and
density fitting on the frozen pair-product aux (RI-J / RI-K)."""

from gs_dft.coulomb.stream import (
    make_pairs, coulomb_energy_fused, coulomb_J_fused,
    full_eri_block, full_eri_block_schur, full_coulomb_energy,
    screened_coulomb_energy,
)
from gs_dft.coulomb.ri import (
    coulomb_block, full_isobra_block, aux_quantities, aux_metric,
    df_coulomb_energy, build_three_center, df_exchange_energy,
    df_coulomb_energy_screened, df_exchange_energy_screened,
)

__all__ = [
    "make_pairs", "coulomb_energy_fused", "coulomb_J_fused",
    "full_eri_block", "full_eri_block_schur", "full_coulomb_energy",
    "screened_coulomb_energy",
    "coulomb_block", "full_isobra_block", "aux_quantities", "aux_metric",
    "df_coulomb_energy", "build_three_center", "df_exchange_energy",
    "df_coulomb_energy_screened", "df_exchange_energy_screened",
]
