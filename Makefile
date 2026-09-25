# Common tasks.
#
#   make figures   rebuild the paper's figures from the committed data in data/figures/
#   make test      run the test suite
#
# The figures use LaTeX text when `latex` is on the PATH and fall back to mathtext otherwise.

PY     ?= uv run python
USETEX ?= $(if $(shell command -v latex 2>/dev/null),1,)
ENV     = TUEPLOTS=1 USETEX=$(USETEX)

# name : module that builds it
FIGS = \
  ablation:experiments.exp1_ablation_ladder.plot_ablation \
  observables:experiments.shared.plot_observables \
  fidelity:experiments.shared.plot_fidelity \
  isoparams:experiments.shared.plot_isoparams

.PHONY: figures test

figures:
	@set -e; for spec in $(FIGS); do \
	  name=$${spec%%:*}; mod=$${spec#*:}; \
	  printf '  %-26s' "$$name"; \
	  $(ENV) $(PY) -m $$mod >/dev/null 2>&1 || { echo 'FAILED'; exit 1; }; echo 'ok'; \
	done; \
	printf '  %-26s' dissociation_ladder_both; \
	MOL=both REL_WIDTH=1.0 $(ENV) $(PY) -m experiments.exp3_lih_dissociation.plot_energy_curve \
	  >/dev/null 2>&1 || { echo 'FAILED'; exit 1; }; echo 'ok'; \
	printf '  %-26s' visual_abstract; \
	$(ENV) $(PY) experiments/shared/visual_abstract.py >/dev/null 2>&1 \
	  || { echo 'FAILED'; exit 1; }; echo 'ok'

test:
	$(PY) -m pytest -q tests
