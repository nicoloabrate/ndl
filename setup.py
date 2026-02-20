"""
Author: N. Abrate.

File: setup.py

Description: Setup for installing rom package.
"""
import setuptools
from setuptools import setup, find_packages

requirements = "requirements.txt"

setup(
   name='NDL',
   version='0.0.3',
   author='N. Abrate',
   author_email='nicolo.abrate@polito.it',
   url='git clone https://github.com/nicoloabrate/ndl',
   package_name = ['NDL'],
   packages = find_packages(),
   license = 'LICENSE.md',
   description = 'NDL generates the ACE files needed by Serpent 2',
   long_description = open('README.md').read(),
   classifiers = ["Programming Language :: Python :: 3",
                  "License :: MIT License",
                  "Operating System :: OS Independent",],
    install_requires=open(requirements).read().splitlines(),
    python_requires = '>=3.3',
)
