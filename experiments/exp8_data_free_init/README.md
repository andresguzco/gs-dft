# Data-free initialization

The initialization ablation of Appendix E: the splat cloud built from the geometry and element
constants alone, adding element exponent ranges, bond-centered splats, bond-aligned anisotropy and
charge-proportional allocation one at a time. `minao_init.py` and `local_init.py` build the
minimal-basis orbital coefficients used for the proteins of Table 3.

```bash
uv run python -m experiments.exp8_data_free_init.run_ladder experiment=exp8_run_ladder system=water
uv run python -m experiments.exp8_data_free_init.plot_curves        # the table
```
