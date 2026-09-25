"""The resource axes are defined once and counted the same way on both sides."""
import numpy as np
import pytest

from experiments.common import resources as R


def test_splat_params_split_into_chart_and_coefficients():
    p = R.splat_params(M=30, n_occ=5)
    assert p["basis"] == 9 * 30                      # 3 center + 3 log-eigenvalue + 3 orientation
    assert p["coeff"] == 30 * 5
    assert p["total"] == p["basis"] + p["coeff"] == 420


def test_quaternion_contributes_three_not_four():
    """A unit quaternion has 3 degrees of freedom; counting its 4 stored numbers inflates every
    splat parameter count by M and biases the whole axis against us."""
    assert R.SPLAT_PARAMS_PER_FN == 9


def test_gto_basis_params_excludes_padding():
    """`exponents` is (n_shells, max_primitives) zero-padded, so `.size` overcounts."""
    class FakeBasis:
        exponents = np.array([[3.0, 1.0, 0.0, 0.0],      # 2 real primitives
                              [5.0, 2.0, 0.5, 0.0],      # 3
                              [1.0, 0.0, 0.0, 0.0]])     # 1
    assert R.gto_basis_params(FakeBasis()) == 2 * 6      # 6 primitives, exponent + coefficient each
    assert R.gto_basis_params(FakeBasis()) != 2 * FakeBasis.exponents.size


def test_both_sides_count_their_basis():
    """The asymmetry that this module exists to prevent."""
    class FakeBasis:
        exponents = np.ones((25, 9))                     # 225 primitives -> 450 basis params
    gto = R.gto_params(FakeBasis(), nao=24, n_occ=5)
    splat = R.splat_params(M=24, n_occ=5)
    assert gto["basis"] > 0, "a Gaussian basis is tabulated, not free"
    assert gto["coeff"] == splat["coeff"], "at matched function count the coefficient blocks match"
    # the honest ratio is set by the per-function basis cost on each side, nothing else
    assert gto["total"] == gto["basis"] + gto["coeff"]


def test_basis_share_falls_as_n_occ_grows():
    """The scale argument: the chart is a per-function cost divided by N_occ, so it amortizes."""
    share = [R.splat_params(M=100, n_occ=n)["basis"] / R.splat_params(M=100, n_occ=n)["coeff"]
             for n in (5, 13, 39)]
    assert share == sorted(share, reverse=True), "basis share must fall with system size"
    assert share[0] == pytest.approx(9 / 5) and share[-1] == pytest.approx(9 / 39)


def test_params_from_row_handles_both_kinds():
    class FakeBasis:
        exponents = np.ones((10, 4))
    assert R.params_from_row({"M": 30}, n_occ=5)["total"] == 420
    assert R.params_from_row({"nao": 24}, n_occ=5, basis_data=FakeBasis())["coeff"] == 120
    assert R.params_from_row({"wall_s": 3}, n_occ=5) is None      # neither size present


def test_breakeven_is_below_one_and_rises_with_system_size():
    """Splats need fewer functions than a Gaussian basis to break even on parameters, and the
    requirement relaxes as the molecule grows."""
    small, large = R.breakeven_fraction(5), R.breakeven_fraction(39)
    assert small < large < 1.0
    assert small == pytest.approx((7.0 + 5) / (9 + 5))


def test_peak_memory_reads_either_spelling():
    """The splat runners write peak_gpu_mb, exp5's Gaussian baselines write peak_mb. A plotter that
    knows only one name renders a splat-vs-GTO memory panel with one representation missing."""
    assert R.peak_mb_from_row({"peak_gpu_mb": "761"}) == 761.0
    assert R.peak_mb_from_row({"peak_mb": "1024.5"}) == 1024.5
    assert R.peak_mb_from_row({"wall_s": "12"}) is None


def test_unavailable_memory_is_none_not_zero():
    """probes.peak_gpu_mb returns -1.0 on CPU or an older JAX. Plotting that as 0 MB would make an
    unmeasured run look like the cheapest point on the panel."""
    assert R.peak_mb_from_row({"peak_gpu_mb": "-1.0"}) is None


def test_training_peak_wins_over_the_whole_job_peak():
    """`peak_gpu_mb` is the job's high-water mark, which on runners that audit after training is set
    by the AUDIT -- a fixed grid cost, not a model cost. Reading it in preference to the clean
    training peak is what made the anion record report 530/531/531 MB across a 3.3x range in M."""
    row = {"peak_train_mb": "203", "peak_gpu_mb": "531"}
    assert R.peak_mb_from_row(row) == 203.0
    assert R.peak_mb_from_row({"peak_gpu_mb": "531"}) == 531.0     # fall back when it is all we have
