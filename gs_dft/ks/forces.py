"""Nuclear forces F_A = −∂E/∂R_A for the Gaussian-splat KS-DFT energy.

Because the splat basis **floats free of the nuclei** (splat centers are variational
parameters, not pinned to atoms), the atom coordinates R enter :class:`SplatKS` in exactly
two places — the electron–nucleus attraction ``E_ext = Tr(P·V_ne(R))`` and the nuclear
repulsion ``E_nn(R)``. Every electronic term (kinetic, Hartree, exchange, XC, the density
itself) depends on R *only* through the converged splat parameters θ*(R). At a variational
minimum ∂E/∂θ* = 0, so by the generalized Hellmann–Feynman theorem the total derivative
collapses to the **explicit partial**:

    F_A = −∂E/∂R_A |_{θ*}  =  −∂/∂R_A [ Tr(P·V_ne(R)) + E_nn(R) ]

i.e. the pure electrostatic Hellmann–Feynman force — and it is **Pulay-free**: the
basis-incompleteness force term that plagues atom-centered GTOs vanishes because the basis
is variationally optimal with respect to its own positions.

Operationally this is just ``-jax.grad`` of the energy w.r.t. ``ks.atom_coords`` with the
splat parameters (``model``) held fixed — one backward pass, no unrolling through the
optimizer. E_nn is always analytic on :class:`SplatKS`, so its force term is always
present; correctness otherwise rests on the inner problem being converged so the implicit
term is negligible (validated in ``experiments/shared/splat_forces_validation.py``).
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from jaxtyping import Float, Array
from gs_dft.ks.energy import SplatKS, SplatModel, SplatState
from gs_dft.ks.orthonormalize import lowdin_orthonormalize, orthonormalize

__all__ = ["splat_forces", "splat_energy_and_forces", "newton_corrected_forces",
           "polish_forces", "df_on_graph_energy"]


def _energy_at_coords(ks, coords, model, state):
    """E_total with the nuclei at ``coords`` and the splats (``model``) held fixed."""
    ks = eqx.tree_at(lambda k: k.atom_coords, ks, coords)
    return ks(model, state)[0]


@eqx.filter_jit
def _grad_at_coords(ks, coords, model, state):
    return jax.grad(_energy_at_coords, argnums=1)(ks, coords, model, state)


@eqx.filter_jit
def _value_and_grad_at_coords(ks, coords, model, state):
    return jax.value_and_grad(_energy_at_coords, argnums=1)(ks, coords, model, state)


def _hf_energy_at_coords(ks, coords, model, state):
    """E_ext(R) + E_nn(R) -- the ONLY two terms of the total energy that depend on the nuclear
    coordinates. Every other term reaches R through the converged splat parameters, which the force
    holds fixed, so ``d/dR`` of this equals ``d/dR`` of the total energy exactly (not approximately)."""
    from gs_dft.ks.energy import _nuclear_repulsion
    from gs_dft.integrals import dense as _full
    from gs_dft import screening as _scr
    occ = model.occupations

    if ks.screen is not None:
        pi, pj, valid = state.pairs
        if ks.mesh is not None:
            from gs_dft.ks import shard as _sharded
            return _sharded.sharded_hf_energy(
                model.basis, model.C, occ, coords, ks.atom_charges, (pi, pj, valid),
                mesh=ks.mesh, rows=_sharded.row_partition(int(model.C.shape[0]), int(ks.mesh.size)),
                skip_pad=ks.screen.skip_pad)
        C_orth = orthonormalize(model.C, model.C.T @ _full.overlap_times(model.basis, model.C))
        return (_scr.screened_external_energy(model.basis, C_orth, occ, pi, pj, coords,
                                              ks.atom_charges, valid, skip_pad=ks.screen.skip_pad)
                + _nuclear_repulsion(coords, ks.atom_charges))
    S, _T, V_ne = _full.one_electron_integrals(model.basis, coords, ks.atom_charges)
    C_orth = lowdin_orthonormalize(model.C, S)
    P = C_orth @ jnp.diag(occ) @ C_orth.T
    return jnp.sum(P * V_ne) + _nuclear_repulsion(coords, ks.atom_charges)


@eqx.filter_jit
def _hf_grad_at_coords(ks, coords, model, state):
    return jax.grad(_hf_energy_at_coords, argnums=1)(ks, coords, model, state)


def splat_forces(
    ks: SplatKS,
    model: SplatModel,
    state: SplatState | None = None,
    *,
    full_graph: bool = False,
) -> Float[Array, "n_atoms 3"]:
    """Hellmann–Feynman nuclear forces F_A = −∂E/∂R_A (Hartree / Bohr).

    Differentiates the total energy w.r.t. ``ks.atom_coords`` only — the splat
    parameters in ``model`` are held fixed, so the result is the explicit partial
    derivative = the converged-geometry Born–Oppenheimer force (exact at the variational
    minimum; the implicit splat-relaxation term is zero by stationarity).

    Differentiates ONLY ``E_ext(R) + E_nn(R)`` (:func:`_hf_energy_at_coords`), the two terms that
    depend on R at all. ``full_graph=True`` differentiates the whole energy instead: the two agree
    to round-off (``tests/unit/test_hf_force_equivalence.py``), but the full graph drags the
    density-fitting backward along and materialises a dense (M, M) metric on one device -- 11.21 GiB
    at 38KNL, which no 80 GB card can hold on top of the working set -- for a contribution that is
    identically zero. Keep it only as a cross-check.
    """
    grad_E = (_grad_at_coords if full_graph else _hf_grad_at_coords)(
        ks, ks.atom_coords, model, state)
    return -grad_E


def splat_energy_and_forces(
    ks: SplatKS,
    model: SplatModel,
    state: SplatState | None = None,
) -> tuple[float, Float[Array, "n_atoms 3"]]:
    """``(E_total, F)`` in one fused value-and-grad pass."""
    E, grad_E = _value_and_grad_at_coords(ks, ks.atom_coords, model, state)
    return float(E), -grad_E


# ---------------------------------------------------------------------------
# Corrected forces. The trained cloud is stationary for the TRAINING functional (frozen DF aux),
# not for the functional whose forces are wanted; the difference is a first-order force term.
# Two ways to close it: one Newton step (linear solve on the training Hessian) or a polish that
# re-converges θ under the target. Both report the net force, which must vanish by translation
# invariance once θ is stationary for the target.
# ---------------------------------------------------------------------------

def _dot(a, b):
    return sum(jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


def _axpy(a, x, y):
    return jax.tree.map(lambda u, v: a * u + v, x, y)


def _n_params(tree):
    return sum(x.size for x in jax.tree.leaves(tree))


def df_on_graph_energy(ks, state):
    """``E(model)`` for a DF ``ks`` with the pair-product aux rebuilt from the splats INSIDE the
    energy, so the aux dependence is differentiated. Metric factored per call (fine up to ~1e4 aux)."""
    from gs_dft.basis.isotropic import init_product_aux
    from gs_dft.coulomb.ri import aux_metric, shift_diag
    term = ks.coulomb

    def energy(model):
        aux = init_product_aux(model.basis, scales=term.scales, diagonal_only=term.diagonal)
        cholV = jax.scipy.linalg.cholesky(shift_diag(aux_metric(aux), term.lam), lower=True)
        st = eqx.tree_at(lambda s: (s.aux, s.cholV), state, (aux, cholV),
                         is_leaf=lambda x: x is None)
        if term.hf_coeff != 0.0:
            st = eqx.tree_at(lambda s: s.aux_K, st, aux, is_leaf=lambda x: x is None)
        return ks(model, st)[0]
    return energy


def _target_energy(target, state, m_st):
    if isinstance(target, SplatKS):
        return lambda tr: target(eqx.combine(tr, m_st), state)[0]
    return lambda tr: target(eqx.combine(tr, m_st))


def _null_projector(m_tr):
    """Remove the exact zero modes: the quaternion norm (per splat) and the C gauge (C -> C T)."""
    qh = m_tr.basis.quat / jnp.linalg.norm(m_tr.basis.quat, axis=1, keepdims=True)
    Cm = m_tr.C
    CtC_inv = jnp.linalg.inv(Cm.T @ Cm)

    def proj(v):
        vq = v.basis.quat - jnp.sum(v.basis.quat * qh, axis=1, keepdims=True) * qh
        vC = v.C - Cm @ (CtC_inv @ (Cm.T @ v.C))
        v = eqx.tree_at(lambda t: t.basis.quat, v, vq)
        return eqx.tree_at(lambda t: t.C, v, vC)
    return proj


def newton_corrected_forces(ks_train, model, state, target, *, cg_iters=50, cg_tol=1e-3,
                            shift_rel=1e-4, precond_probes=16, fd_eps=1e-6, verbose=True):
    """``(F, F_HF, info)``: forces of ``target``'s relaxed surface for a model trained on ``ks_train``.

    F = F_HF + ∇_R[λ · ∇_θ E_target] with H_train λ = ∇_θ E_target(θ*), exact to second order in the
    target gradient. ``target`` is a ``SplatKS`` (e.g. exact Coulomb) or any ``E(model) -> scalar``.
    Hessian-vector products are central differences of the training gradient (the kernels' custom
    VJPs admit neither forward mode nor a second reverse pass); the solve is CG with the zero modes
    projected out and a Jacobi preconditioner from ``precond_probes`` Hutchinson probes.
    """
    from gs_dft.ks.train import _partition
    m_tr, m_st = _partition(model)
    coords0 = ks_train.atom_coords
    E_target = _target_energy(target, state, m_st)
    n = _n_params(m_tr)

    def e_train(tr, coords):
        ks_ = eqx.tree_at(lambda k: k.atom_coords, ks_train, coords)
        return ks_(eqx.combine(tr, m_st), state)[0]

    g_theta = eqx.filter_jit(lambda tr: jax.grad(lambda t: e_train(t, coords0))(tr))
    g_R = eqx.filter_jit(lambda tr: jax.grad(lambda c: e_train(tr, c))(coords0))

    def fd(fn, v):
        s = fd_eps / float(np.sqrt(float(_dot(v, v)) / n))
        return jax.tree.map(lambda a, b: (a - b) / (2 * s), fn(_axpy(s, v, m_tr)), fn(_axpy(-s, v, m_tr)))

    proj = _null_projector(m_tr)
    F_HF = -g_R(m_tr)
    r = proj(jax.grad(E_target)(m_tr))
    rr = float(_dot(r, r))
    mu = shift_rel * float(_dot(r, fd(g_theta, r))) / rr
    A = lambda v: proj(_axpy(mu, v, fd(g_theta, proj(v))))

    # Jacobi preconditioner: Hutchinson estimate of diag(H), floored
    diag = jax.tree.map(jnp.zeros_like, m_tr)
    for k in jax.random.split(jax.random.PRNGKey(0), precond_probes):
        ks_ = jax.tree.unflatten(jax.tree.structure(m_tr), list(jax.random.split(k, len(jax.tree.leaves(m_tr)))))
        z = jax.tree.map(lambda x, kk: jax.random.rademacher(kk, x.shape, float), m_tr, ks_)
        diag = jax.tree.map(lambda d, zz, h: d + zz * h / precond_probes, diag, z, fd(g_theta, z))
    floor = 1e-3 * max(float(jnp.max(jnp.abs(x))) for x in jax.tree.leaves(diag))
    Minv = jax.tree.map(lambda d: 1.0 / jnp.maximum(jnp.abs(d), floor), diag)
    prec = lambda v: proj(jax.tree.map(lambda m_, a: m_ * a, Minv, v))

    x = jax.tree.map(jnp.zeros_like, r)
    res, z = r, prec(r)
    p, rz = z, float(_dot(r, z))
    rel, k = 1.0, 0
    for k in range(1, cg_iters + 1):
        Ap = A(p)
        pAp = float(_dot(p, Ap))
        if pAp <= 0:                                  # negative curvature: stop with the current x
            if verbose:
                print(f"  cg {k:3d}  negative curvature, stopping", flush=True)
            break
        alpha = rz / pAp
        x = _axpy(alpha, p, x)
        res = _axpy(-alpha, Ap, res)
        rel = float(np.sqrt(float(_dot(res, res)) / rr))
        if verbose:
            print(f"  cg {k:3d}  rel res {rel:.3e}", flush=True)
        if rel < cg_tol:
            break
        z = prec(res)
        rz_new = float(_dot(res, z))
        p = _axpy(rz_new / rz, p, z)
        rz = rz_new
    F = F_HF + fd(g_R, x)                             # ∇_R[λ · ∇_θ E] = ∂_R∂_θE · λ
    info = {"g_target_rms": float(np.sqrt(rr / n)), "cg_iters": k, "cg_rel_res": rel, "shift": mu,
            "n_grad": 2 * (k + 2 + precond_probes) + 2}
    return F, F_HF, info


def polish_forces(ks_train, model, state, target, *, steps=200, lr=3e-4, method="adam",
                  every=25, verbose=True):
    """``(F, model, info)``: re-converge θ under ``target`` from the trained model, then take the
    Hellmann-Feynman force there. ``method`` is ``adam`` (constant lr, clip 1) or ``lbfgs``.
    ``info["trace"]`` holds ``(step, E, |g|rms, |net F|)`` at the report cadence."""
    import optax
    from gs_dft.ks.train import _partition
    m_tr, m_st = _partition(model)
    coords0 = ks_train.atom_coords
    E_target = _target_energy(target, state, m_st)
    n = _n_params(m_tr)
    E = eqx.filter_jit(E_target)
    VG = eqx.filter_jit(jax.value_and_grad(E_target))
    F_of = eqx.filter_jit(lambda tr: -jax.grad(
        lambda c: eqx.tree_at(lambda k: k.atom_coords, ks_train, c)(eqx.combine(tr, m_st), state)[0])(coords0))

    def report(k, tr, e, g):
        F = F_of(tr)
        grms = float(np.sqrt(float(_dot(g, g)) / n))
        net = float(jnp.linalg.norm(jnp.sum(F, axis=0)))
        if verbose:
            print(f"  polish {k:4d}  E={float(e):.8f}  |g|rms={grms:.3e}  |net F|={net:.3e}", flush=True)
        return (k, float(e), grms, net)

    tr = m_tr
    if method == "lbfgs":
        opt = optax.lbfgs(memory_size=20)
        vg = optax.value_and_grad_from_state(E)
    else:
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    st = opt.init(tr)
    e, g = VG(tr)
    trace = [report(0, tr, e, g)]
    for k in range(1, steps + 1):
        if method == "lbfgs":
            e, g = vg(tr, state=st)
            u, st = opt.update(g, st, tr, value=e, grad=g, value_fn=E)
        else:
            e, g = VG(tr)
            u, st = opt.update(g, st, tr)
        tr = eqx.apply_updates(tr, u)
        if k % every == 0 or k == steps:
            trace.append(report(k, tr, *VG(tr)))
    F = F_of(tr)
    return F, eqx.combine(tr, m_st), {"trace": trace, "n_grad": steps + len(trace)}
