# PyCalib
Python library for classifier calibration


## Unittest

```
python3.6 -m unittest discover pycalib/
```

## Install as a submodule

Create a folder in your project in which to place your submodules

```
mkdir lib
```

Then add this package as a submodule

```
git submodule add git submodule add git@github.com:perellonieto/PyCalib.git lib/PyCalib
```

then install into your virtual environment

```
pip install -e lib/PyCalib
```

If you want to use your current virtual environment in a Jupyter notebook you
can create a new kernel with

```
python -m ipykernel install --user --name NameOfThisKernel --display-name
"NameOfThisKernel"
```
