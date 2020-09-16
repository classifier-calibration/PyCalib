from distutils.core import setup
from distutils.util import convert_path

with open("README.md", 'r') as f:
    long_description = f.read()

main_ns = {}
ver_path = convert_path('pycalib/__init__.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
  name = 'pycalib',
  packages = ['pycalib'],
  install_requires=[
    'numpy',
    'scipy',
    'sklearn',
    'matplotlib',
  ],
  version=main_ns['__version__'],
  description = 'Python library with tools for classifier calibration.',
  author = 'Miquel Perello Nieto, Hao Song, Telmo de Menezes e Silva Filho',
  author_email = 'perello.nieto@gmail.com',
  url = 'https://github.com/perellonieto/PyCalib',
  download_url = 'https://github.com/perellonieto/pycalib/archive/{}.tar.gz'.format(main_ns['__version__']),
  keywords = ['classifier calibration', 'calibration', 'classification'],
  classifiers = [],
  long_description_content_type='text/markdown',
  long_description=long_description
)
