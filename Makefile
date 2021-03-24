.PHONY: venv

pip:
	pip install --upgrade pip

venv:
	python3.8 -m venv venv

requirements: pip
	pip install -r requirements.txt

requirements-dev: requirements pip
	pip install -r requirements-dev.txt

test: requirements-dev
	pytest --doctest-modules --cov-report=term-missing --cov=pycalib pycalib

check-readme: requirements-dev
	twine check dist/*

build: requirements-dev
	python3.8 setup.py sdist

pypi: build check-readme
	twine upload dist/*

doc: requirements-dev
	cd docs; make clean; make html

# From Scikit-learn
code-analysis:
	flake8 pycalib | grep -v external
	pylint -E pycalib/ -d E1103,E0611,E1101

clean:
	rm -rf ./dist
