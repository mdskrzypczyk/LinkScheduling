PYTHON        = python3
PIP           = pip3
PYTEST        = pytest
SOURCEDIR     = jobscheduling
SIMULATIONDIR = simulations
TESTS         = tests

clean: _clean_pyc _clean_dist

_clean_pyc:
	@find . -name '*.pyc' -delete

_clean_dist:
	@rm -f dist/*

build: _clean_dist
	@$(PYTHON) setup.py sdist bdist_wheel

install: verify build
	@$(PIP) install dist/*.whl

lint:
	@$(PYTHON) -m flake8 $(SIMULATIONDIR) $(SOURCEDIR) $(TESTS)

python-deps:
	@$(PIP) install -r requirements.txt

tests:
	@$(PYTEST)

verify: clean python-deps lint tests

.PHONY: clean lint python-deps tests verify
