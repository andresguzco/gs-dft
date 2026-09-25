"""The ``RESULT`` line contract: emit and parse round-trip, ``kind`` comes first, and partial lines
never satisfy the resume guard.
"""
import re

import pytest

from experiments.common.results import result, partial, parse, iter_results, GUARD, jsonl_writer


def test_result_roundtrip_all_kinds():
    for kind in ("splat", "gto", "ckpt", "bench", "train", "obs", "forces"):
        line = result(kind, "water", M=96, E=-76.33664720, seed=0, converged=True)
        d = parse(line)
        assert d["_tag"] == "RESULT" and d["kind"] == kind and d["system"] == "water"
        assert d["M"] == "96" and d["seed"] == "0" and d["converged"] == "True"
        assert float(d["E"]) == pytest.approx(-76.33664720)


def test_kind_is_the_first_key():
    # the `grep "^RESULT kind=ckpt"` resume contract depends on this ordering
    line = result("ckpt", "ethanol", path="a.eqx", E=-154.9)
    assert line.startswith("RESULT kind=ckpt system=ethanol ")


def test_float_and_bool_formatting():
    # energies are floats → fixed 8-decimal; counts (MB, steps, nao) are passed as ints → bare
    assert "E=-76.33664720" in result("splat", "water", E=-76.33664720)
    assert "peak_gpu_mb=1096" in result("splat", "water", peak_gpu_mb=round(1096.0))
    assert "conv=True" in result("gto", "water", conv=True)
    assert "conv=False" in result("gto", "water", conv=False)   # bool not stringified as 0/1


def test_tiny_diagnostic_floats_survive():
    # a ~1e-16 residual must not flatten to 0.00000000 under the energy-oriented %.8f
    line = result("gto", "water", mu_grid_vs_analytic=1.2e-16, E=-76.3)
    d = parse(line)
    assert float(d["mu_grid_vs_analytic"]) == pytest.approx(1.2e-16, rel=1e-2)
    assert d["mu_grid_vs_analytic"] != "0.00000000"
    assert parse(result("gto", "w", x=0.0))["x"] == "0.00000000"   # exact zero stays plain


def test_whitespace_value_is_rejected():
    # a value with whitespace would break the split-on-space parser — must fail at emit
    with pytest.raises(ValueError):
        result("gto", "water", note="two words")


def test_partial_never_matches_the_resume_guard():
    # a RESULT_PARTIAL line must NOT satisfy `^RESULT ` (else a clipped run reads as done)
    pl = partial("splat", "water", step=200, E=-76.1, wall_s=30)
    assert pl.startswith("RESULT_PARTIAL ")
    assert re.search(GUARD, pl) is None
    assert re.search(GUARD, result("splat", "water", E=-76.1)) is not None


def test_parse_ignores_non_result_lines():
    assert parse("# a comment") is None
    assert parse("") is None
    assert parse("crossed E_DZ at step 100") is None


def test_parse_accepts_result_family_tags():
    # RESULT_FLOP (count_flops) parses order-independently and never trips the resume guard
    d = parse("RESULT_FLOP system=water M=96 per_outer=1.2e9")
    assert d["_tag"] == "RESULT_FLOP" and d["M"] == "96" and d["per_outer"] == "1.2e9"
    assert re.search(GUARD, "RESULT_FLOP system=water M=96") is None


def test_iter_results_filters_by_tag_and_kind(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\n".join([
        "# header",
        result("gto", "f_anion", basis="cc-pvdz", E=-99.66568),
        result("splat", "f_anion", M=28, E=-99.7),
        partial("splat", "f_anion", step=100, E=-99.5),
    ]) + "\n")
    gto = list(iter_results(str(log), kind="gto"))
    assert len(gto) == 1 and gto[0]["basis"] == "cc-pvdz"
    partials = list(iter_results(str(log), tag="RESULT_PARTIAL"))
    assert len(partials) == 1 and partials[0]["step"] == "100"
    assert len(list(iter_results(str(log)))) == 2       # both RESULT lines, partial excluded


def test_jsonl_writer_truncates_then_appends(tmp_path):
    p = str(tmp_path / "run.jsonl")
    with jsonl_writer(p) as w:
        w.write({"step": 0, "E": -1.0})
    with jsonl_writer(p, resume=True) as w:             # resume appends, keeps history
        w.write({"step": 1, "E": -2.0})
    lines = open(p).read().splitlines()
    assert len(lines) == 2 and lines[0].startswith('{"step": 0')
    with jsonl_writer(p) as w:                          # no resume truncates
        w.write({"step": 0, "E": -3.0})
    assert len(open(p).read().splitlines()) == 1


def test_csv_records_read_back_as_the_parsed_lines(tmp_path):
    """`write_csv` then `iter_results` yields the same rows, in the same order, as parsing the logs."""
    from experiments.common.results import read_records, write_csv
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    a.write_text(result("splat", "water", M=24, E=-76.1) + "\n"
                 + partial("splat", "water", step=10, E=-75.0) + "\n"
                 + "TRACE rung=P3_floor step=5 t=1.500 E=-76.00000000\n"
                 + "  step   10  E=-75.000000  |g|=1.00e+00  λmin=2.00e-01\n"
                 + "unrelated output\n")
    b.write_text(result("gto", "water", basis="cc-pvdz", E=-76.2) + "\n")
    out = tmp_path / "data.csv"
    assert write_csv([str(a), str(b)], str(out)) == 5
    strip = lambda d: {k: v for k, v in d.items() if k not in ("_source", "_record")}
    assert [strip(d) for d in iter_results(str(out))] == list(iter_results([str(a), str(b)]))
    assert [strip(d) for d in iter_results(str(out), tag="RESULT_PARTIAL")] == \
        list(iter_results([str(a)], tag="RESULT_PARTIAL"))
    step = next(read_records(str(out), record="STEP"))
    assert step["step"] == "10" and step["grad_norm"] == "1.00e+00" and step["lam_min"] == "2.00e-01"
    assert [d["_source"] for d in read_records(str(out), sources=["b.log"])] == ["b.log"]
