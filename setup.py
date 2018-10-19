#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


requirements = [
    "pyyaml",
    "future",
    "numpy",
    "pandas",
    "tqdm",
    "related>=0.6.0",
    "enum34",
    "colorlog",
    "jinja2",
    "cookiecutter",
    # sometimes required
    "h5py",
    "urllib3>=1.21.1,<1.23",
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
    "tensorflow"
]

setup(
    name='kipoi',
    version='0.5.2',
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
