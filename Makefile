.PHONY: help guide calibration analytic hydrogenic hydrogen-exact atomic \
	potentials h2 h2-grid fig-calibration fig-gp93-building-block \
	fig-hydrogen-regularization fig-atomic-systematics fig-he-levels \
	fig-he-comparison fig-radial-potentials fig-h2 fig-all validate \
	postprocess integration

help:
	@echo "Step-by-step reproduction (run targets in the documented order):"
	@echo "  make calibration                 data for calibration"
	@echo "  make fig-calibration             one calibration figure"
	@echo "  make analytic                    analytic GP93/H data"
	@echo "  make fig-gp93-building-block     one GP93 figure"
	@echo "  make fig-hydrogen-regularization one H-regularization figure"
	@echo "  make hydrogenic                  hydrogenic table data"
	@echo "  make hydrogen-exact              unregularized H SCF data"
	@echo "  make atomic                      atom and He-basis data"
	@echo "  make potentials                  radial-potential data"
	@echo "  make h2 h2-grid                  H2 curve and grid-check data"
	@echo "  make <fig-...>                   one figure from saved data"
	@echo "  make validate                    cross-file scientific checks"
	@echo "Paper-table values are direct CSV outputs; see results/README.md."
	@echo "See REPRODUCTION_GUIDE.md before running a target."

guide:
	@echo "Open REPRODUCTION_GUIDE.md and results/README.md."

calibration:
	python calculations/calibration/run_calibration.py

analytic:
	python calculations/analytic/run_analytic.py

hydrogenic:
	python calculations/analytic/run_hydrogenic.py

hydrogen-exact:
	python calculations/atomic/run_hydrogen_exact.py

atomic:
	python calculations/atomic/run_atomic.py

potentials:
	python calculations/potentials/run_potentials.py

h2:
	python calculations/h2/run_h2.py

h2-grid:
	python calculations/h2/validate_grid.py

fig-calibration:
	python calculations/plot_results.py fig-calibration

fig-gp93-building-block:
	python calculations/plot_results.py fig-gp93-building-block

fig-hydrogen-regularization:
	python calculations/plot_results.py fig-hydrogen-regularization

fig-atomic-systematics:
	python calculations/plot_results.py fig-atomic-systematics

fig-he-levels:
	python calculations/plot_results.py fig-he-levels

fig-he-comparison:
	python calculations/plot_results.py fig-he-comparison

fig-radial-potentials:
	python calculations/plot_results.py fig-radial-potentials

fig-h2:
	python calculations/plot_results.py fig-h2

fig-all:
	python calculations/plot_results.py all

validate:
	python calculations/validate_results.py

postprocess:
	python reproduce.py --stage postprocess

integration:
	python reproduce.py
