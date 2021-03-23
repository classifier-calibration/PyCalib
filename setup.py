from distutils.util import convert_path
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

main_ns = {}
ver_path = convert_path('pycalib/__init__.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
  name = 'pycalib',
  packages = find_packages(exclude=['tests.*', 'tests', 'docs.*', 'docs']),
  install_requires=[
    'numpy',
    'scipy',
    'scikit-learn',
    'matplotlib',
    'statsmodels'
  ],
  version=main_ns['__version__'],
  description = 'Python library with tools for classifier calibration.',
  author = 'Miquel Perello Nieto, Hao Song, Telmo de Menezes e Silva Filho',
  author_email = 'perello.nieto@gmail.com',
  url = 'https://www.perellonieto.com/PyCalib/index.html',
  download_url = 'https://github.com/perellonieto/pycalib/archive/{}.tar.gz'.format(main_ns['__version__']),
  keywords = ['classifier calibration', 'calibration', 'classification'],
  classifiers = [],
  long_description=long_description,
  long_description_content_type='text/markdown'
)
