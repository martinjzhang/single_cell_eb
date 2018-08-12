from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call
from distutils.command.build_py import build_py as _build_py
import sys, os.path

setup(name='sceb',
      version='0.2',
      description='Single Cell Empirical Bayes: empirical Bayes estimators for distributional quantities of single cell gene distribution.',
      url='https://github.com/martinjzhang/single_cell_eb',
      author='Martin Zhang',
      author_email='jinye@stanford.edu',
      license='Stanford University',
      packages=['sceb'],
      zip_safe=False)