"""Matplotlib and tueplots styling for the paper figures.

``TUEPLOTS=1`` switches on the ICLR sizes and fonts and ``USETEX=1`` LaTeX text::

    from experiments.common.style import apply
    usetex = apply(nrows=1, ncols=len(systems))   # True iff LaTeX text is on

Without ``TUEPLOTS``, ``apply`` does nothing and the plot uses matplotlib defaults.
"""
import os


#: ICLR text width, inches — what `figsizes.iclr20XX` returns for `rel_width=1.0`.
TEXT_WIDTH_IN = 5.5

PAPER_FIG_HEIGHT_IN = 1.28


def ratio_for_height(height_in, ncols=1):
    """`height_to_width_ratio` that yields `height_in` inches, whatever `ncols` is."""
    return float(height_in) * int(ncols) / TEXT_WIDTH_IN


def apply(height_in=None, **figsize_kw):
    """Update ``plt.rcParams`` for paper mode when ``TUEPLOTS`` is set; return whether LaTeX text is
    active (``USETEX``), which callers use to pick TeX vs plain axis labels.

    ``figsize_kw`` is passed straight to the tueplots ``figsizes`` bundle — the plotters vary in what
    they need (``nrows=/ncols=``, ``rel_width=``, ``height_to_width_ratio=``); default is a single
    panel. Called with no TUEPLOTS set, it is a no-op that just reports USETEX."""
    import matplotlib.pyplot as plt

    usetex = bool(os.environ.get("USETEX"))
    if not os.environ.get("TUEPLOTS"):
        return usetex
    from tueplots import bundles, figsizes
    bundle = getattr(bundles, "iclr2024", getattr(bundles, "iclr2023", None))
    fsizes = getattr(figsizes, "iclr2024", getattr(figsizes, "iclr2023", None))
    plt.rcParams.update(bundle(usetex=usetex, family="serif"))
    if usetex:
        plt.rcParams["text.latex.preamble"] = (
            r"\usepackage[version=4]{mhchem}" "\n" r"\usepackage{amsmath}" "\n"
            r"\usepackage{amssymb}")
    if not usetex:
        import matplotlib.font_manager as _fm
        have = {f.name for f in _fm.fontManager.ttflist}
        for cand in ("STIXGeneral", "DejaVu Serif"):
            if cand in have:
                plt.rcParams["font.serif"] = [cand] + list(plt.rcParams.get("font.serif", []))
                plt.rcParams["mathtext.fontset"] = "stix" if cand == "STIXGeneral" else "dejavuserif"
                break
    kw = dict(figsize_kw or {"nrows": 1, "ncols": 1})
    if height_in is not None:
        kw["height_to_width_ratio"] = ratio_for_height(height_in, kw.get("ncols", 1))
    plt.rcParams.update(fsizes(**kw))
    return usetex

def save_at_height(fig, out, height_in=None, tol=0.01, iters=6, **savefig_kw):
    """Save `fig` so the RESULTING FILE is `height_in` inches tall.

    The canvas is adjusted until the tight bounding box has the target aspect ratio, then saved
    tight. Labels and legends do not scale with the canvas, so this takes two or three passes."""
    height_in = PAPER_FIG_HEIGHT_IN if height_in is None else float(height_in)
    target = height_in / TEXT_WIDTH_IN
    for _ in range(iters):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        if bb is None or bb.height <= 0 or bb.width <= 0:
            break
        if abs(bb.height / bb.width - target) < tol / TEXT_WIDTH_IN:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w, max(h * target / (bb.height / bb.width), 0.2))
    savefig_kw.setdefault("bbox_inches", "tight")
    fig.savefig(out, **savefig_kw)
