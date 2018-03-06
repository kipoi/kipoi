#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError, OSError):
    # print("Unable to convert REAMDE.md to rst using pypandoc")
    long_description = open('README.md').read()


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
    "scikit-learn",
    "cython",
    # "genomelake",
    "keras",
    "tensorflow"
]

setup(
    name='kipoi',
    version='0.2.5',
    description="Kipoi",  # TODO - update the description
    author="Kipoi team",
    author_email='avsec@in.tum.de',
    url='https://github.com/kipoi/kipoi',
    long_description="Kipoi",  # TODO - update the description
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "develop": test_requirements,
        # variant effect prediction
        "vep": ["pyvcf",
                "cyvcf2",
                "pybedtools",
                "pysam",  # required by pybedtools
                "intervaltree",
                "deepdish",
                ],
    },
    entry_points={'console_scripts': ['kipoi = kipoi.__main__:main']},
    license="MIT license",
    zip_safe=False,
    keywords=["model zoo", "deep learning",
              "computational biology", "bioinformatics", "genomics"],
    test_suite='tests',
    # package_data={'kipoi': ['logging.conf']},
    include_package_data=True,
    tests_require=test_requirements
)
