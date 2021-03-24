[![CI][ci:b]][ci]
[![Documentation][documentation:b]][documentation]
[![License BSD3][license:b]][license]
![Python3.8][python:b]
[![pypi][pypi:b]][pypi]
[![codecov][codecov:b]][codecov]

[ci]: https://github.com/perellonieto/PyCalib/actions/workflows/ci.yml
[ci:b]: https://github.com/perellonieto/pycalib/workflows/CI/badge.svg
[documentation]: https://github.com/perellonieto/PyCalib/actions/workflows/documentation.yml
[documentation:b]: https://github.com/perellonieto/pycalib/workflows/Documentation/badge.svg
[license]: https://github.com/perellonieto/PyCalib/blob/master/LICENSE.txt
[license:b]: https://img.shields.io/github/license/perellonieto/pycalib.svg
[python:b]: https://img.shields.io/badge/python-3.8-blue
[pypi]: https://badge.fury.io/py/pycalib
[pypi:b]: https://badge.fury.io/py/pycalib.svg
[codecov]: https://codecov.io/gh/perellonieto/PyCalib
[codecov:b]: https://codecov.io/gh/perellonieto/PyCalib/branch/master/graph/badge.svg?token=AYMZPLELT3

PyCalib
=======
Python library for classifier calibration

Development
===========

There is a make file to automate some of the common tasks during development.
After downloading the repository create the virtual environment with the
command

```
make venv
```

This will create a `venv` folder in your current folder. The environment needs
to be loaded out of the makefile with

```
source venv/bin/activate
```

After the environment is loaded, all dependencies can be installed with

```
make requirements-dev
```

Unittest
--------

Run the unittest with the command

```
make unittest
```

The test will show a unittest result including the coverage of the code.
Ideally we want to increase the coverage to cover most of the library.

Contiunous Integration
----------------------

Every time a commit is pushed to the master branch a unittest is run following
the workflow [.github/workflows/ci.yml](.github/workflows/ci.yml). The CI badge
in the README file will show if the test has passed or not.

Documentation
-------------

All documentation is done ussing the [Sphinx documentation
generator][sphinx:l].  The documentation is written in
[reStructuredText][rst:l] (\*.rst) files in the `docs/source` folder. The
examples with images in folder [docs/source/examples](docs/source/examples) are
generated automatically with [Sphinx-gallery][sphinx:g] from the python code in
folder [examples/](examples/) starting with `xmpl_{example_name}.py`.

[rst:l]: https://docutils.sourceforge.io/rst.html
[sphinx:l]: https://www.sphinx-doc.org/en/master/
[sphinx:g]: https://sphinx-gallery.github.io/stable/index.html

The documentation has its own Makefile inside folder [docs](docs).

To build the documentation go into the docs folder, clean the old documentation
with

```
make clean
```

And build the new documentation with

```
make html
```

After building the documentation, a new folder should appear in `docs/build/`
with an `index.html` that can be opened locally for further exploration.

The documentation is always build and deployed every time a new commit is
pushed to the master branch with the workflow
[.github/workflows/documentation.yml](.github/workflows/documentation.yml).

Check Readme
------------

It is possible to check that the README file passes some tests for Pypi by
running

```
make check-readme
```

Upload to PyPi
--------------

After testing that the code passes all unittests and upgrading the version in
the file `pycalib/__init__.py` the code can be published in Pypi with the
following command:

```
make pypi
```

It may require user and password if these are not set in your home directory a
file  __.pypirc__

```
[pypi]
username = __token__
password = pypi-yourtoken
```

Contributors
------------

This code has been adapted by Miquel from several previous codes. The following
is a list of people that has been involved in some parts of the code.

- Miquel Perello Nieto
- Hao Song
- Telmo Silva Filho
- Markus Kängsepp
