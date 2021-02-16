.. -*- mode: rst -*-

PyCalib
=======
Python library for classifier calibration


Unittest
--------

    python3.6 -m unittest discover pycalib/

Install as a submodule
----------------------

Create a folder in your project in which to place your submodules

    mkdir lib

Then add this package as a submodule

    git submodule add git@github.com:perellonieto/PyCalib.git lib/PyCalib

then install into your virtual environment

    pip install -e lib/PyCalib

If you want to use your current virtual environment in a Jupyter notebook you
can create a new kernel with

    python -m ipykernel install --user --name NameOfThisKernel --display-name "NameOfThisKernel"

Upload to PyPi
--------------

Create the files to distribute

    python3.6 setup.py sdist

Ensure twine is installed

    pip install twine

Upload the distribution files

    twine upload dist/*

It may require user and password if these are not set in your home directory a
file  __.pypirc__

    [pypi]
    username = __token__
    password = pypi-yourtoken

Check the README file for Pypi
------------------------------

In order to check that the readme file is compliant with Pypi standards,
install the following Python package

    pip install readme-renderer

and run the following command

    twine check dist/*

Contributors
------------

This code has been adapted by Miquel from several previous codes. The following
is a list of people that has been involved in some parts of the code.

- Miquel Perello Nieto
- Hao Song
- Telmo Silva Filho
- Markus Kängsepp
