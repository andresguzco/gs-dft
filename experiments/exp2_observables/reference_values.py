"""Experimental reference values with their provenance.

1. A microwave Stark measurement gives the dipole of the vibrational ground state (mu_0); a
   clamped-nuclei calculation gives the equilibrium dipole (mu_e). For water they differ by 7.3 mD,
   so the equilibrium value is used where one exists, and ``preferred`` says which.
2. Ethanol's measured dipole depends on the rotamer (gauche: 1.679 D). ``conformer`` records what
   was measured, and a comparison must match it.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DIPOLES", "ExperimentalValue", "check_comparable"]


@dataclass(frozen=True)
class ExperimentalValue:
    value: float                 # in the stated unit, AS MEASURED
    unit: str
    uncertainty: float | None    # as published; None when the source states none
    kind: str                    # "mu_0" (ground-state, ZPV-averaged) | "mu_e" (equilibrium)
    conformer: str | None        # the rotamer measured, when it matters
    source: str
    note: str = ""
    preferred: bool = False      # the right comparator for a clamped-nuclei calculation
    clamped_nuclei_offset_mD: float | None = None
    offset_source: str = ""

    @property
    def clamped_nuclei_target(self) -> float | None:
        """`value` corrected to what a nonrelativistic clamped-nuclei calculation should reproduce."""
        if self.clamped_nuclei_offset_mD is None:
            return None
        return self.value + self.clamped_nuclei_offset_mD * 1e-3


DIPOLES: dict[str, tuple[ExperimentalValue, ...]] = {
    "water": (
        ExperimentalValue(
            1.85498, "D", 0.00009, "mu_0", None,
            "Shostak, Ebenstein & Muenter, J. Chem. Phys. 94, 5875-5882 (1991), "
            "doi:10.1063/1.460471 — molecular-beam electric resonance, (000) state",
            note="THE MEASUREMENT, and the most precise one: a factor ~7 tighter than Clough 1973 "
                 "and consistent with Lovas (1978) 1.854 D and Gregory et al., Science 275, 814 "
                 "(1997) 1.855 D. It is mu_0, so it is not directly what a clamped-nuclei "
                 "calculation produces; `clamped_nuclei_offset_mD` carries that bridge.",
            clamped_nuclei_offset_mD=+4.1,
            offset_source=(
                "Lodi et al., J. Chem. Phys. 128, 044304 (2008), doi:10.1063/1.2817606, Table VIII: "
                "nonrelativistic Born-Oppenheimer mu_e = 0.7310(5) a.u., relativistic correction "
                "-0.0017, vibrational averaging +0.0001, giving mu_0 = 0.7294(6). Our calculation "
                "is nonrelativistic and clamped-nuclei, so the comparator is mu_0 + (0.7310-0.7294) "
                "a.u. = +0.0016 a.u. = +4.1 mD, i.e. 1.8591 D. Lodi's own mu_e converts to 1.8580 D; "
                "the ~1 mD difference is the theory-experiment gap in mu_0 itself and is inside the "
                "combined uncertainties."),
            preferred=True),
        ExperimentalValue(
            1.8546, "D", 0.0006, "mu_0", None,
            "Clough, Beers, Klein & Rothman, J. Chem. Phys. 59, 2254 (1973), doi:10.1063/1.1680328 "
            "— Stark measurements on H2O, HDO and D2O",
            note="The widely quoted value; superseded in precision by Shostak 1991."),
        ExperimentalValue(
            1.8473, "D", 0.0010, "mu_e", None,
            "Clough, Beers, Klein & Rothman, J. Chem. Phys. 59, 2254 (1973), doi:10.1063/1.1680328 "
            "— equilibrium value from the same Stark analysis",
            note="RECORDED, NOT USED, AND THIS FILE PREVIOUSLY PREFERRED IT. It places mu_e 7.3 mD "
                 "BELOW mu_0, but Shostak's vibrationless constant (1.857 D) and Lodi's CCSD(T)/CBS "
                 "mu_e (1.8580 D) both place mu_e AT OR ABOVE mu_0. The sign of the 1973 vibrational "
                 "correction is contradicted by both, and using it moves the target by 11 mD against "
                 "a deficit this experiment reports at ~30 mD."),
    ),
    "ethanol": (
        ExperimentalValue(
            1.441, "D", 0.007, "mu_0", "anti",
            "Takano, Sasada & Satoh, J. Mol. Spectrosc. 26, 157-162 (1968), "
            "doi:10.1016/0022-2852(68)90159-8 — microwave Stark, 8-34 GHz, b-type",
            note="ANTI/trans rotamer of the parent CH3CH2OH; components mu_a=0.046(14), "
                 "mu_b=1.438(19), mu_c=0 by Cs symmetry. This is the comparator for our geometry "
                 "(H-O-C-C dihedral 180 deg). THE LITERATURE SPREAD IS 1.44-1.52 D AND MUST BE "
                 "QUOTED: the JPL catalogue lists mu_b=1.462 unexplained, and NIST CCCBDB lists "
                 "1.520 D while printing 'Value of 1.441 D for CH3CH2OH seems low' — but that 1.520 "
                 "is the DEUTERATED CH3CHDOH isotopologue, and a total dipole is isotope-invariant "
                 "to ~0.01 D, so the 0.08 D gap is unresolved rather than a correction. Sourced "
                 "from the abstract plus two peer-reviewed secondary citations (Mueller et al., "
                 "A&A 587, A92 (2016); TMC-1 detection, A&A 2023); the publisher returns 403 to "
                 "automated retrieval, so the PDF itself has not been read.",
            preferred=True),
        ExperimentalValue(
            1.679, "D", None, "mu_0", "gauche",
            "Kakar & Quade, J. Chem. Phys. 72, 4300-4307 (1980), doi:10.1063/1.439723; "
            "components mu_a=1.264(10), mu_b=0.104(8), mu_c=1.101(16)",
            note="GAUCHE rotamer. Not the comparator for our anti geometry."),
        ExperimentalValue(
            1.520, "D", None, "mu_0", None,
            "Hellwege & Hellwege, Landolt-Bornstein (1974); via NIST CCCBDB",
            note="Reported for CH3CHDOH; the source itself flags the 1.441 D CH3CH2OH value as "
                 "'seems low'. Recorded for completeness, NOT for quoting."),
    ),
}


def check_comparable(system: str, conformer: str | None = None):
    """The preferred experimental dipole for ``system``, or a loud refusal.

    Raises rather than returning a best guess: a silently mismatched conformer or a ground-state
    value substituted for an equilibrium one produces a plausible table that is wrong by more than
    the effect being reported.
    """
    vals = DIPOLES.get(system)
    if not vals:
        raise KeyError(f"no experimental dipole recorded for {system!r}; add one with its source "
                       f"to DIPOLES rather than inlining a number")
    if conformer is not None:
        match = [v for v in vals if v.conformer == conformer]
        if match:
            return match[0]
        known = sorted({v.conformer for v in vals if v.conformer}) or ["none recorded"]
        raise ValueError(
            f"{system}: the calculation is the {conformer!r} rotamer but the recorded measurements "
            f"are for {known}. These are different molecules for this observable; add the "
            f"{conformer!r} value WITH its source rather than substituting a neighbour.")
    conf_specific = [v for v in vals if v.conformer is not None]
    if conf_specific:
        raise ValueError(
            f"{system}: the experimental values are conformer-specific "
            f"({sorted({v.conformer for v in conf_specific})}); pass the computed rotamer "
            f"explicitly so a mismatch cannot pass silently.")
    pref = [v for v in vals if v.preferred]
    if len(pref) != 1:
        raise ValueError(f"{system}: expected exactly one preferred value, got {len(pref)}")
    return pref[0]
