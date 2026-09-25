"""Compute the splat observables of Figure 3 from a converged checkpoint and compare them with the
Gaussian ladder.

Observables: the exact-Coulomb energy, Hellmann-Feynman nuclear forces, dipole and traceless
quadrupole, KS orbital energies from one warm ``scf_solve`` at the converged basis, the kinetic
energy, the density at the nuclei, and density errors against each Gaussian rung. Everything is
evaluated on the ``obs_grid_level`` grid, finer than the training grid and shared with
``gto_ref.py``.

    uv run python -m experiments.exp2_observables.observables experiment=exp2_observables system=water

Run ``train_ckpt.py`` and the ``gto_ref.py`` rungs first.
"""
import glob
import os

import hydra
from omegaconf import DictConfig
import numpy as np
import jax.numpy as jnp
import equinox as eqx
from dftax.grid import becke_grid

from experiments.common import systems, results, tracking, builders
from gs_dft.integrals import dense as fullmod
from gs_dft.ks.energy import SplatKS, SplatState
from gs_dft.ks.terms import df
from gs_dft.ks.train import evaluate, _GRID_LEVELS
from gs_dft.ks.forces import splat_forces
from gs_dft.ks.orthonormalize import lowdin_orthonormalize
from experiments.exp2_observables.scf import scf_solve
from experiments.exp2_observables.train_ckpt import RES, fresh_init, ckpt_path, M_BY
from experiments.exp2_observables.gto_ref import multipoles, AU2DEBYE
from dftax.energy.xc import PBE

from experiments.exp2_observables.reference_values import check_comparable
ORDER = ["cc-pvdz", "cc-pvtz", "cc-pvqz", "cc-pv5z"]
AU2EV = 27.211386245988


def _rotamer(mol) -> str | None:
    """The H-O-C-C rotamer of an alcohol, measured from the geometry, or None if not an alcohol.

    An alcohol's experimental dipole is conformer-specific and the rotamers differ by more than this
    experiment's error bar, so the comparator cannot be chosen by assumption.
    """
    syms = [s.upper() for s in mol.symbols]
    xyz = np.asarray(mol.atom_coords(), dtype=float)
    O = [i for i, s in enumerate(syms) if s == "O"]
    Cs = [i for i, s in enumerate(syms) if s == "C"]
    Hs = [i for i, s in enumerate(syms) if s == "H"]
    if len(O) != 1 or len(Cs) < 2 or not Hs:
        return None
    O = O[0]
    HO = min(Hs, key=lambda h: np.linalg.norm(xyz[h] - xyz[O]))
    C1 = min(Cs, key=lambda c: np.linalg.norm(xyz[c] - xyz[O]))
    C2 = max((c for c in Cs if c != C1), key=lambda c: np.linalg.norm(xyz[c] - xyz[O]))
    b0, b1, b2 = xyz[HO] - xyz[O], xyz[C1] - xyz[O], xyz[C2] - xyz[C1]
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - np.dot(b0, b1) * b1, b2 - np.dot(b2, b1) * b1
    d = abs(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))
    return "anti" if abs(d - 180.0) < 40 else ("gauche" if 20 < d < 120 else None)


def _experimental_dipole(system, mol):
    """``(value, None)`` when a comparable measurement exists, else ``(None, reason)``.

    Reported rather than raised: the experimental comparison is one column and every other
    observable stays valid without it -- but it is not substituted with the nearest available
    number either.
    """
    try:
        return check_comparable(system, conformer=_rotamer(mol)), None
    except (KeyError, ValueError) as exc:
        return None, str(exc).replace("\n", " ")[:160]


def _wf_archive(system):
    """``(kind, basis, npz)`` of the wavefunction reference: the largest-``nao`` ``cc_*`` archive,
    else the largest ``mp2_*`` one; None when neither exists."""
    for kind in ("cc", "mp2"):
        files = sorted(glob.glob(os.path.join(RES, f"{kind}_{system}_*.npz")),
                       key=lambda f: int(np.load(f, allow_pickle=True)["nao"]))
        if files:
            return kind, os.path.basename(files[-1])[len(f"{kind}_{system}_"):-4], np.load(files[-1], allow_pickle=True)
    return None


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    system = cfg.system
    M_ckpt = int(cfg.m) if cfg.get("m", None) is not None else M_BY[system]
    tracking.init(cfg, __name__, name=f"obs_{cfg.system}_M{M_ckpt}")
    mol = systems.molecule(cfg)
    model_t, aux_t = fresh_init(system, mol, M_ckpt)
    _polished = ckpt_path(system, M_ckpt, polished=True)
    # Prefer the polished checkpoint: the splat force is the explicit partial at fixed splats, so
    # it is only the true force once theta is stationary. Never mix -- an energy from the raw
    # checkpoint beside a force from the polished one describes no single state.
    _path = _polished if os.path.exists(_polished) else ckpt_path(system, M_ckpt)
    _state = "polished" if _path == _polished else "raw"
    print(f"reading {_state} checkpoint: {os.path.basename(_path)}", flush=True)
    model, aux = eqx.tree_deserialise_leaves(_path, (model_t, aux_t))
    coords = jnp.array(mol.atom_coords())
    charges = jnp.array(mol.atom_charges(), float)
    occ = model.occupations
    n_occ = occ.shape[0]

    # The basis-independent REPORTING grid: identical points to gto_ref's saved rho, which is what
    # makes the density L1 grid-exact. Deliberately finer than the training grid.
    nr, lb = _GRID_LEVELS[int(cfg.obs_grid_level)]
    gp, gw = becke_grid(list(mol.symbols), coords, n_radial=nr, lebedev=lb)


    ef = SplatKS((coords, charges), PBE(), grid=(gp, gw), coulomb=df())
    E = evaluate(ef, model)
    S, T_mat, _ = fullmod.one_electron_integrals(model.basis, coords, charges)
    C = lowdin_orthonormalize(model.C, S)
    P = C @ jnp.diag(occ) @ C.T
    T = float(jnp.sum(P * T_mat))
    rho = np.asarray(fullmod.eval_density_on_grid(model.basis, C, occ, gp))
    rho_nuc = np.asarray(fullmod.eval_density_on_grid(model.basis, C, occ, coords))
    N_e = float(np.dot(np.asarray(gw), rho))
    mu, theta = multipoles(np.asarray(gp), np.asarray(gw), rho,
                           np.asarray(coords), np.asarray(charges))
    F = np.asarray(splat_forces(ef, model, SplatState(aux=aux)))
    res = scf_solve(model.basis, aux, coords, charges, gp, gw, occ, PBE(),
                    C0=C, max_iter=50)
    eps = np.asarray(res.eps)
    homo, lumo = eps[n_occ - 1], eps[n_occ]

    M = int(model.C.shape[0])
    print(f"=== exp4 observables: {system} (splat full-cov M={M}) ===")
    print(f"  E_exact={E:.8f}  N_e(grid)={N_e:.6f} (expect {float(occ.sum()):.1f})")
    print(f"  scf(eps): n_iter={res.n_iter} conv={res.converged}")
    mu_D = float(np.linalg.norm(mu) * AU2DEBYE)
    exp_val, exp_why = _experimental_dipole(system, mol)
    if exp_val is not None:
        tgt = exp_val.clamped_nuclei_target
        print(f"  dipole mu_D={mu_D:.4f}  exp_D={exp_val.value:.5f} ({exp_val.kind}"
              f"{', ' + exp_val.conformer if exp_val.conformer else ''})"
              f"  delta_vs_measured={1e3 * (mu_D - exp_val.value):+.1f} mD"
              + (f"  target={tgt:.5f} (+{exp_val.clamped_nuclei_offset_mD:.1f} mD corr)"
                 f"  delta_vs_target={1e3 * (mu_D - tgt):+.1f} mD" if tgt is not None
                 else "  [no clamped-nuclei correction recorded]"))
        print(f"    measurement: {exp_val.source}")
        if exp_val.offset_source:
            print(f"    correction : {exp_val.offset_source}")
    else:
        print(f"  dipole mu_D={mu_D:.4f}  exp_D=UNAVAILABLE ({exp_why})")
    print(f"  spectrum homo={homo:.5f} lumo={lumo:.5f} gap_eV={(lumo - homo) * AU2EV:.4f}  "
          f"kinetic T={T:.6f}")
    print(f"  rho@nuclei " + " ".join(f"{v:.4f}" for v in rho_nuc))

    refs = {}
    for b in ORDER:
        path = f"{RES}/ref_{system}_{b}.npz"
        if os.path.exists(path):
            refs[b] = np.load(path)
    if not refs:
        print("(no GTO refs yet — emitting the splat-side row; re-run after gto_ref.py for the "
              "comparison fields)")
        _gf = builders.grid_fields(mol, PBE(), model, cfg.grid_check, float(E))
        print(results.result("obs", system, M=M, vs="none", E=E, **_gf, mu_D=mu_D,
                            exp_mu_D=(exp_val.value if exp_val else float("nan")),
                            exp_kind=(exp_val.kind if exp_val else "none"),
                            exp_conformer=(exp_val.conformer or "n/a" if exp_val else "n/a"),
                            d_exp_mD=((mu_D - exp_val.value) * 1e3 if exp_val else float("nan")),
                            exp_target_D=((exp_val.clamped_nuclei_target or float("nan"))
                                          if exp_val else float("nan")),
                            d_target_mD=((mu_D - exp_val.clamped_nuclei_target) * 1e3
                                         if exp_val and exp_val.clamped_nuclei_target
                                         else float("nan")),
                            homo=float(homo), lumo=float(lumo), gap=float(lumo - homo), T=T,
                            rho_nuc=",".join(f"{v:.4f}" for v in rho_nuc),
                            maxF=float(np.abs(F).max())), flush=True)
        return
    best = ORDER[max(i for i, b in enumerate(ORDER) if b in refs)]
    rb = refs[best]
    ext_grad, ext_src = {}, {}
    for b in ORDER:
        gp = f"{RES}/refgrad_{system}_{b}.npz"
        if b in refs and not np.isfinite(refs[b]["grad"]).all() and os.path.exists(gp):
            z = np.load(gp)
            if np.isfinite(z["grad"]).all():
                ext_grad[b], ext_src[b] = np.asarray(z["grad"]), str(z["src"])
    def _grad_of(b):
        return ext_grad[b] if b in ext_grad else refs[b]["grad"]

    # The force reference is the largest rung that actually HAS a force: gto_ref skips it above
    # `force_max_nao`, and at a geometry that is not a PBE minimum the true force is nonzero, so a
    # residual force is not an error and "larger" would not mean "worse".
    with_force = [b for b in ORDER if b in refs and np.isfinite(_grad_of(b)).all()]
    fbest = with_force[-1] if with_force else None
    fb = {"grad": _grad_of(fbest)} if fbest else None
    fsrc = ext_src.get(fbest, "dftax") if fbest else "none"
    if fbest and fbest in ext_grad:
        print(f"  force reference: {fbest} from {fsrc} (dftax has none at any cardinal)", flush=True)

    hdr = f"  {'quantity':28s}  splat" + "".join(f" | {b[-3:]:>9s}" for b in refs)
    print("\n" + hdr)
    def row(name, sv, key, fn=lambda r: float(r), fmt="9.4f"):
        cells = "".join(f" | {fn(refs[b][key]):{fmt}}" for b in refs)
        print(f"  {name:28s}  {sv:{fmt}}{cells}")
    row("E (Ha)", E, "E", fmt="12.6f")
    row("mu (D)", float(np.linalg.norm(mu) * AU2DEBYE), "mu",
        fn=lambda v: float(np.linalg.norm(v) * AU2DEBYE))
    row("Theta_zz (au)", float(theta[2, 2]), "theta", fn=lambda v: float(v[2, 2]))
    row("<T> (Ha)", T, "T", fmt="12.6f")
    row("eps_HOMO", float(homo), "mo_energy", fn=lambda v: float(v[n_occ - 1]), fmt="9.5f")
    row("eps_LUMO", float(lumo), "mo_energy", fn=lambda v: float(v[n_occ]), fmt="9.5f")
    row("KS gap (eV)", float((lumo - homo) * AU2EV), "mo_energy",
        fn=lambda v: float((v[n_occ] - v[n_occ - 1]) * AU2EV), fmt="9.4f")
    row("max|F| (Ha/Bohr)", float(np.abs(F).max()), "grad",
        fn=lambda v: float(np.abs(v).max()), fmt="9.5f")

    mad_occ = float(np.abs(eps[:n_occ] - rb["mo_energy"][:n_occ]).mean())
    dF = (float(np.abs(F - (-fb["grad"])).max()) if fb is not None else float("nan"))
    w = np.asarray(gw)
    _sref = f"{RES}/splatref_{system}_M{M}.npz"
    _tmp = _sref + f".tmp{os.getpid()}"
    with open(_tmp, "wb") as _fh:
        np.savez(_fh, rho=rho, grad=-F, w=w, E=float(E), M=int(M), n_grid=int(rho.shape[0]))
    os.replace(_tmp, _sref)
    print(f"  archived this rung to {_sref}", flush=True)
    l1 = {b: float(np.dot(w, np.abs(rho - refs[b]["rho"]))) for b in refs}
    floor = (float(np.dot(w, np.abs(refs["cc-pvqz"]["rho"] - rb["rho"])))
             if "cc-pvqz" in refs and best == "cc-pv5z" else float("nan"))
    # Wavefunction reference for the density and force panels: the largest CCSD(T) archive, else
    # the largest MP2 one (the dipeptide has no CC). The GTO rungs are scored against the same
    # archive here, so the figure's two ladders share one reference.
    cc_fields = {}
    wf = _wf_archive(system)
    if wf is not None:
        kind, basis, _d = wf
        cc_fields.update(wf_kind=kind, wf_basis=basis,
                         L1_rho_wf=float(np.dot(w, np.abs(rho - np.asarray(_d["rho"])))))
        if np.isfinite(_d["grad"]).all() and _d["grad"].shape == F.shape:
            cc_fields["dF_wf"] = float(np.abs(F - (-np.asarray(_d["grad"]))).max())
        print(f"  vs {kind.upper()}/{basis}:  density L1 = {cc_fields['L1_rho_wf']:.4f} e"
              + (f"   max|F_splat - F_wf| = {cc_fields['dF_wf']:.5f} Ha/Bohr"
                 if "dF_wf" in cc_fields else ""), flush=True)
        for b in refs:
            gb = _grad_of(b)
            dF_b = (float(np.abs(gb - np.asarray(_d["grad"])).max())
                    if np.isfinite(gb).all() and gb.shape == _d["grad"].shape else float("nan"))
            print(results.result("gto_wf", system, basis=b, wf_kind=kind, wf_basis=basis,
                                 L1_rho_wf=float(np.dot(w, np.abs(refs[b]["rho"] - np.asarray(_d["rho"])))),
                                 dF_wf=dF_b), flush=True)

    print(f"\n  vs {best}:  occ-spectrum MAD = {mad_occ * 1e3:.2f} mHa   "
          f"max|F_splat - F_ref| = {dF:.5f} Ha/Bohr")
    print("  density L1 = " + "  ".join(f"{b[-3:]}:{v:.4f}" for b, v in l1.items()) +
          f"   (QZ-5Z floor: {floor:.4f}) e")
    _gf = builders.grid_fields(mol, PBE(), model, cfg.grid_check, float(E))
    print(results.result("obs", system, M=M, vs=best, ckpt=_state, E=E, **_gf,
                        mu_D=mu_D,
                        exp_mu_D=(exp_val.value if exp_val else float("nan")),
                        exp_kind=(exp_val.kind if exp_val else "none"),
                        exp_conformer=(exp_val.conformer or "n/a" if exp_val else "n/a"),
                        d_exp_mD=((mu_D - exp_val.value) * 1e3 if exp_val else float("nan")),
                        exp_target_D=((exp_val.clamped_nuclei_target or float("nan"))
                                      if exp_val else float("nan")),
                        d_target_mD=((mu_D - exp_val.clamped_nuclei_target) * 1e3
                                     if exp_val and exp_val.clamped_nuclei_target else float("nan")),
                        homo=float(homo), lumo=float(lumo), gap=float(lumo - homo), T=T,
                        rho_nuc=",".join(f"{v:.4f}" for v in rho_nuc),
                        mad_occ_mha=float(mad_occ * 1e3), dF_max=float(dF),
                        dF_vs=(fbest or "none"), dF_src=fsrc,
                        maxF=float(np.abs(F).max()),
                        L1_rho=float(l1[best]), L1_floor=float(floor), **cc_fields), flush=True)
    for a in range(F.shape[0]):
        print(f"  F[{a}] splat=({F[a][0]:+.5f},{F[a][1]:+.5f},{F[a][2]:+.5f})  "
              f"{best}=({-rb['grad'][a][0]:+.5f},{-rb['grad'][a][1]:+.5f},{-rb['grad'][a][2]:+.5f})")


if __name__ == "__main__":
    main()
