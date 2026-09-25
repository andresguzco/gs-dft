"""The optimizer comparison of Appendix E: every applicable Optax optimizer on alanine dipeptide.

Each optimizer finds its own learning rate inside its job, in two phases:

    A. range test (about 600 steps): ramp the rate exponentially from 1e-6 to 1 and record the
       energy; lr* = argmin(E) / RANGE_SAFETY.
    B. production (`steps`): restart from the same PRNG key and run the standard schedule (clip,
       200-step warmup, cosine to 0.01 lr*) at peak lr*.

The gradient is full-batch and deterministic, so the phase-A curve has no sampling noise.
``lr_star`` and ``lr_diverge`` are both written to the RESULT row.

Weight decay is off for every optimizer, including those whose Optax default turns it on, except the
one ``adamw_wd`` arm: decay adds a term that is not in the energy. Optimizers built for min-max
problems, for memory-bound models or for stochastic gradients are not included.

    uv run python -m experiments.exp1_ablation_ladder.optimizers experiment=exp1_optsweep opt=adam
"""
import inspect
import math
import time

import hydra
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from omegaconf import DictConfig

from gs_dft import init_model, nao as nao_of
from gs_dft.ks.energy import SplatKS
from gs_dft.ks.terms import df, pairlist
from gs_dft.ks.train import evaluate, monitor, native_grid, train
from experiments.common import builders, results, systems, tracking
from experiments.exp1_ablation_ladder.machinery import (EXACT_EVAL_MAX_ATOMS, EXACT_EVAL_MAX_M,
                                                        _free_mb, _peak_mb, electron_count,
                                                        kinetic_energy)

RANGE_STEPS = 600            # ~100 points per decade over the six decades below
RANGE_LO, RANGE_HI = 1e-6, 1e0
RANGE_SAFETY = 3.0          # lr* = lr_diverge / this. A decade below divergence is the usual rule;
                             # `lr_min` and `lr_diverge` are recorded so another factor needs no re-run.
CLIP = 1.0                   # the shipped global-norm clip, held FIXED across every arm: it is a
                             # stabilizer, not part of the optimizer, so it must not vary with one.


def _factory(name, **fixed):
    """An optax alias as ``schedule -> GradientTransformation``, with weight decay forced off."""
    fn = getattr(optax, name)
    params = inspect.signature(fn).parameters
    if "weight_decay" in params and "weight_decay" not in fixed:
        fixed["weight_decay"] = 0.0
    return lambda sched: fn(sched, **fixed)


# arm -> (factory, family). The family is what the appendix groups by; a leaderboard of 21 names
# says less than six groups with a mechanism attached to each.
ARMS = {
    # per-coordinate adaptive, first + second moment
    "adam":      (_factory("adam"),      "adam-family"),
    "adamax":    (_factory("adamax"),    "adam-family"),
    "nadam":     (_factory("nadam"),     "adam-family"),
    "radam":     (_factory("radam"),     "adam-family"),
    "amsgrad":   (_factory("amsgrad"),   "adam-family"),
    "adabelief": (_factory("adabelief"), "adam-family"),
    "yogi":      (_factory("yogi"),      "adam-family"),
    "adan":      (_factory("adan"),      "adam-family"),
    "novograd":  (_factory("novograd"),  "adam-family"),
    # second moment only
    "adagrad":   (_factory("adagrad"),   "second-moment"),
    "rmsprop":   (_factory("rmsprop"),   "second-moment"),
    "adadelta":  (_factory("adadelta"),  "second-moment"),
    # trust ratio / layerwise
    "lars":      (_factory("lars"),      "trust-ratio"),
    "lamb":      (_factory("lamb"),      "trust-ratio"),
    "fromage":   (_factory("fromage"),   "trust-ratio"),
    # sign based — the endpoints of the segment `signum` sits inside
    "lion":      (_factory("lion"),      "sign"),
    "sign_sgd":  (_factory("sign_sgd"),  "sign"),
    # designed for full-batch deterministic optimization, which is exactly our setting
    "rprop":     (_factory("rprop"),     "full-batch"),
    # the control: no per-coordinate adaptation at all
    "sgd":       (_factory("sgd", momentum=0.9, nesterov=True), "non-adaptive"),
    # the one deliberate weight-decay arm — §3's claim, tested rather than asserted
    "adamw_wd":  (_factory("adamw", weight_decay=1e-4), "weight-decay"),
}
LBFGS = "lbfgs"


def _ramp(lo=RANGE_LO, hi=RANGE_HI, n=RANGE_STEPS):
    """Exponential learning-rate ramp: lo at step 0, hi at step n."""
    ratio = hi / lo
    return lambda t: lo * ratio ** (jnp.minimum(t, n) / n)


def lr_at(step, lo=RANGE_LO, hi=RANGE_HI, n=RANGE_STEPS):
    return lo * (hi / lo) ** (min(step, n) / n)


def pick_lr(steps, energies):
    """(lr_star, lr_min, lr_diverge, why) from the phase-A curve."""
    e = np.asarray(energies, dtype=float)
    s_ = np.asarray(steps, dtype=int)
    keep = (s_ > 0) & np.isfinite(e)
    if not keep.any():
        return float("nan"), float("nan"), float("nan"), "no finite energy after the priming frame"
    e, s_ = e[keep], s_[keep]

    i_min = int(np.argmin(e))
    lr_min = lr_at(int(s_[i_min]))
    lr_star = lr_min / RANGE_SAFETY

    # For the record only: where the curve first gives back 5% of what it had gained past the
    # minimum. Reported so a different rule can be applied later without re-running anything.
    lr_div, why = float("nan"), f"min at step {int(s_[i_min])}"
    tail = e[i_min:]
    if tail.size > 1:
        gain = max(float(e[0] - e[i_min]), 1e-9)
        over = np.nonzero(tail > e[i_min] + 0.05 * gain)[0]
        if over.size:
            lr_div = lr_at(int(s_[i_min + int(over[0])]))
            why += f", gave back 5% by {lr_div:.2e}"
    if i_min >= len(e) - 2:
        why += " (AT THE TOP OF THE RAMP — the useful band may lie above it)"
    return lr_star, lr_min, lr_div, why


def build(cfg):
    """The production KS + a fresh model, identical to what `machinery` builds."""
    mol = systems.molecule(cfg)
    M = int(cfg.m) if cfg.m is not None else int(cfg.m_mult * nao_of(mol))
    model = init_model(mol, M, jr.PRNGKey(int(cfg.seed)))
    ks = SplatKS(mol, builders.xc_of(cfg.xc),
                 grid=native_grid(mol, int(cfg.grid_level), chunk=4096),
                 coulomb=df(lam=float(cfg.df_lam), scales=builders.aux_scales(cfg.aux_mult)),
                 screen=pairlist(eps=1e-7, pad=float(cfg.screen_pad)))
    return mol, M, model, ks


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    arm = str(cfg.opt)
    tracking.init(cfg, __name__, name=f"opt_{arm}_{cfg.system}_s{cfg.seed}")
    mol, M, model, ks = build(cfg)
    n_occ = int(mol.nelectron) // 2
    print(f"# exp1 optimizer sweep: arm={arm} system={cfg.system} M={M} n_occ={n_occ} "
          f"steps={cfg.steps} seed={cfg.seed}", flush=True)

    t0 = time.time()
    row = dict(arm=arm, family=ARMS.get(arm, (None, "quasi-newton"))[1], M=M, n_occ=n_occ,
               steps=int(cfg.steps), seed=int(cfg.seed))
    failed = "none"

    # ---- phase A: the range test -----------------------------------------------------------
    lr_star, lr_min, lr_div, why = (float("nan"),) * 3 + ("skipped",)
    if arm != LBFGS:
        ramp_steps, ramp_E = [], []

        def _ramp_cb(step, _m, _a, E):
            ramp_steps.append(int(step))
            ramp_E.append(float(E))
            print(f"RANGE arm={arm} step={int(step)} lr={lr_at(int(step)):.3e} "
                  f"E={float(E):.8f}", flush=True)

        try:
            opt_a = optax.chain(optax.clip_by_global_norm(CLIP), ARMS[arm][0](_ramp()))
            train(ks, model, steps=RANGE_STEPS, refresh_every=int(cfg.refresh), monitor=None,
                  optimizer=opt_a, snapshot_every=1, snapshot_cb=_ramp_cb)
        except Exception as exc:                                   # noqa: BLE001
            print(f"# range test raised ({type(exc).__name__}): {str(exc)[:140]}", flush=True)
        lr_star, lr_min, lr_div, why = pick_lr(ramp_steps, ramp_E)
        print(f"# phase A: lr_star={lr_star:.3e} lr_min={lr_min:.3e} lr_diverge={lr_div:.3e} ({why})", flush=True)

    # ---- reset, exactly: same key, same cloud ----------------------------------------------
    _, _, model, _ = build(cfg)

    # ---- phase B: the shipped schedule at lr* ----------------------------------------------
    E = E_train = e_kin = gnorm = nelec = float("nan")
    n_target = 2.0 * n_occ
    try:
        if arm == LBFGS:
            opt_b = optax.lbfgs(linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=20))
        else:
            if not math.isfinite(lr_star) or lr_star <= 0:
                raise ValueError(f"phase A produced no usable rate ({why})")
            w = min(int(cfg.steps) // 10, 200)
            sched = optax.join_schedules(
                [optax.linear_schedule(lr_star * 0.01, lr_star, w),
                 optax.cosine_decay_schedule(lr_star, max(1, int(cfg.steps) - w), alpha=0.01)], [w])
            opt_b = optax.chain(optax.clip_by_global_norm(CLIP), ARMS[arm][0](sched))

        trace_n = max(int(cfg.steps) // 300, 1)

        def _trace(step, model_, _a, Ev):
            try:
                nel = electron_count(model_)
            except Exception:                                      # noqa: BLE001
                nel = float("nan")
            print(f"TRACE rung=opt_{arm} step={int(step)} t={time.time() - t0:.3f} "
                  f"E={float(Ev):.8f} nelec={nel:.6f} dnelec={nel - n_target:+.3e}", flush=True)

        res = train(ks, model, steps=int(cfg.steps), refresh_every=int(cfg.refresh),
                    monitor=monitor(max(int(cfg.steps) // 10, 1)), optimizer=opt_b,
                    snapshot_every=trace_n, snapshot_cb=_trace)
        gnorm = float(res.grad_norm[-1]) if len(res.grad_norm) else float("nan")
        E_train = float(ks(res.model, res.state)[0])
        E = (float(evaluate(ks, res.model, res.state))
             if (len(mol.symbols) <= EXACT_EVAL_MAX_ATOMS and M <= EXACT_EVAL_MAX_M)
             else float("nan"))
        e_kin = kinetic_energy(res.model)
        nelec = electron_count(res.model)
    except Exception as exc:                                       # noqa: BLE001
        failed = ("oom" if "RESOURCE_EXHAUSTED" in str(exc) or "Out of memory" in str(exc)
                  else type(exc).__name__)
        print(f"# {arm} FAILED ({failed}): {str(exc)[:180]}", flush=True)

    wall = time.time() - t0
    row.update(lr_star=lr_star, lr_min=lr_min, lr_diverge=lr_div, E=E, E_train=E_train,
               df_residual=(E_train - E), E_kin=e_kin, grad_norm=gnorm,
               nelec=nelec, nelec_err=(nelec - n_target),
               nelec_ok=bool(abs(nelec - n_target) < 1e-3 * n_target),
               kinetic_ok=bool(e_kin >= 0.0), failed=failed,
               peak_train_mb=_peak_mb(), gpu_free_mb=_free_mb(),
               wall_s=round(wall, 1), ms_step=round(1e3 * wall / max(int(cfg.steps), 1), 2))
    print(results.result("opt", cfg.system, **row), flush=True)


if __name__ == "__main__":
    main()
