.PHONY: venv

pip:
	pip install --upgrade pip

venv:
	python3.8 -m venv venv

requirements: pip
	pip install -r requirements.txt

requirements-dev: requirements pip
	pip install -r requirements-dev.txt

unittest: requirements-dev
	pytest --cov-report=term-missing --cov=pycalib pycalib

check-readme: requirements-dev
	twine check dist/*

build: requirements-dev
	python3.8 setup.py sdist

pypi: build check-readme
	twine upload dist/*

clean:
	rm -rf ./dist
