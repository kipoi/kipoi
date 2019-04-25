#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


requirements = [
    "pyyaml",
    "future",
    "numpy",
    "pandas>=0.21.0",
    "tqdm",
    "attrs>=17.4.0",
    "related>=0.6.0",
    "enum34",
    "colorlog",
    "jinja2",
    "cookiecutter>=1.6.0",
    # sometimes required
    "h5py",
    "urllib3>=1.21.1", #,<1.23",
    "tinydb",
    "kipoi-utils>=0.1.12",
    "kipoi-conda>=0.1.6",
    "deprecation>=2.0.6"
]

test_requirements = [
    "bumpversion",
    "wheel",
    "jedi",
    "epc",
    "pytest>=3.3.1",
    "pytest-xdist",  # running tests in parallel
    "pytest-pep8",  # see https://github.com/kipoi/kipoi/issues/91
    "pytest-cov",
    "coveralls",
    "scikit-learn",
    "cython",
    "keras",
    "tensorflow",
    #"genomelake>=0.1.4",     # test_10_KipoiModel.py (fails on circle-ci)
    "zarr>=2.2.0",            # test_161_writers.py
    #"cyvcf2>=0.10.0",        # test_20_cli_examples.py (and others) (fails on circle-ci)
    "kipoi-interpret>=0.1.2", # test_42_kipoi_interpret.py
    "concise>=0.6.6"
]

setup(
    name='kipoi',
    version='0.6.12',
    description="Kipoi: model zoo for genomics",
    author="Kipoi team",
    author_email='avsec@in.tum.de',
    url='https://github.com/kipoi/kipoi',
    long_description="Kipoi: model zoo for genomics. http://kipoi.org/",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "develop": test_requirements,
        "vep": ["kipoi_veff"],  # backcompatibily
    },
    entry_points={'console_scripts': ['kipoi = kipoi.__main__:main']},
    license="MIT license",
    zip_safe=False,
    keywords=["model zoo", "deep learning",
              "computational biology", "bioinformatics", "genomics"],
    test_suite='tests',
    include_package_data=True,
    tests_require=test_requirements
)
