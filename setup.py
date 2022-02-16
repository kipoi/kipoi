#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

from setuptools import setup, find_packages
from setuptools.command.install import install


# this will be overwritten by bumpversion
version = '0.8.0'

requirements = [
    "pyyaml",
    "future",
    "numpy",
    "pandas>=0.21.0",
    "tqdm",
    "attrs>=17.4.0",
    "related>=0.6.0",
    "colorlog",
    "jinja2",
    "cookiecutter>=1.6.0",
    # sometimes required
    "h5py",
    "urllib3>=1.21.1",  # ,<1.23",
    "tinydb",
    "kipoi-utils>=0.3.8",
    "kipoi-conda>=0.1.6",
    "deprecation>=2.0.6"
]

test_requirements = [
    "bump2version",
    "gitpython",
    "wheel",
    "jedi",
    "epc",
    "pyarrow",
    "pytest>=3.3.1",
    "pytest-xdist",  # running tests in parallel
    "pytest-pep8",  # see https://github.com/kipoi/kipoi/issues/91
    "pytest-cov",
    "coveralls",
    "scikit-learn",
    "cython",
    "keras",
    "tensorflow",
    "zarr>=2.2.0",  # test_161_writers.py
    # "cyvcf2>=0.10.0",        # test_20_cli_examples.py (and others) (fails on circle-ci)
    "kipoi-interpret>=0.1.2",  # test_42_kipoi_interpret.py
    "concise>=0.6.6" # Necessary for tests under folder tests/legacy-keras. Does not work for python>=3.8
]

setup(
    name='kipoi',
    version=version,
    description="Kipoi: model zoo for genomics",
    author="Kipoi team",
    author_email='avsec@in.tum.de',
    url='https://github.com/kipoi/kipoi',
    long_description="Kipoi: model zoo for genomics. http://kipoi.org/",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "develop": test_requirements,
    },
    entry_points={'console_scripts': ['kipoi = kipoi.__main__:main']},
    license="MIT license",
    zip_safe=False,
    keywords=["model zoo", "deep learning",
              "computational biology", "bioinformatics", "genomics"],
    test_suite='tests',
    include_package_data=True,
    tests_require=test_requirements,
)
