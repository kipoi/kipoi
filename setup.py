#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    # TODO?
    # Convert .md -> .rst for nicer display on pypi
    # https://stackoverflow.com/questions/26737222/pypi-description-markdown-doesnt-work
    readme = readme_file.read()

requirements = [
    "pyyaml",
    "future",
    "numpy",
    "pandas",
    "tqdm",
    "related>=0.6.0",
    "enum34",
    "colorlog",
    "cookiecutter",
    # sometimes required
    "h5py",
    "deepdish",
    # only required for score_variants
    "pyvcf",
    "cyvcf2",
    "pybedtools",
    "pysam",  # required by pybedtools
]

test_requirements = [
    "pytest>=3.3.1",
    "virtualenv",
    # model requirements - TODO - add also pytorch
    "scikit-learn",
    "cython",
    "genomelake",
    "keras",
    "tensorflow",
]

setup(
    name='kipoi',
    version='0.0.1',
    description="Kipoi model-zoo command-line tool",
    author="Kipoi team",  # whom to put here?
    author_email='...',
    url='https://github.com/kipoi/model-zoo',
    long_description=readme,
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "develop": ["bumpversion",
                    "wheel",
                    "jedi",
                    "epc",
                    "pytest>=3.3.1",
                    "pytest-xdist",  # running tests in parallel
                    "pytest-pep8",  # see https://github.com/kipoi/kipoi/issues/91
                    "pytest-cov"
                    ],
    },
    entry_points={'console_scripts': ['kipoi = kipoi.__main__:main']},
    license="MIT license",
    zip_safe=False,
    keywords=["model zoo", "deep learning",
              "computational biology", "bioinformatics", "genomics"],
    test_suite='tests',
    package_data={'kipoi': ['logging.conf']},
    include_package_data=True,
    tests_require=test_requirements
)
