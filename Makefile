unittest:
	pytest --cov=pycalib pycalib

check-readme:
	twine check dist/*
