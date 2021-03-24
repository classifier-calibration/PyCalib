unittest:
	pytest --cov-report=term-missing --cov=pycalib pycalib

check-readme:
	twine check dist/*
