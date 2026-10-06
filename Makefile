PYTHON ?= python
ENV = PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg

.PHONY: smoke test verify-checkpoints

smoke:
	$(ENV) $(PYTHON) -m pytest tests/test_phase_executor.py tests/test_experiment_authoring.py -q -p no:cacheprovider

test:
	$(ENV) $(PYTHON) -m pytest tests -q -p no:cacheprovider

verify-checkpoints:
	$(PYTHON) pretrained/atari/checkpoints/download.py --verify-only

.PHONY: reproduce check-data test-statistics package

reproduce:
	$(ENV) $(PYTHON) reproduction/plots/reproduce.py --check --output-dir build/paper

check-data:
	$(ENV) $(PYTHON) reproduction/validation/check.py

test-statistics:
	$(ENV) $(PYTHON) -m unittest discover -s tests -p test_statistical_comparisons.py

package:
	$(PYTHON) support/package.py --output dist/stackelberg-pomdp-jair-2026-v1.0.0.tar.gz
