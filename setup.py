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
    "h5py",
    "numpy",
    "pandas",
    "keras",
    "tqdm",
    "deepdish",
    "related==0.4",
    "enum34",
    # "pytorch"
]

# TODO - don't explicitly require pytorch
#        - how can we get the DatasetLoader functionality?


test_requirements = [
    "pytest",
    "virtualenv",
]
# TODO - require conda to be installed? - to create custom environments


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
                    "pytest",
                    "pytest-pep8",
                    "pytest-cov"],
    },
    entry_points={'console_scripts': ['kipoi = kipoi.__main__:main']},
    license="MIT license",
    zip_safe=False,
    keywords=["model zoo", "deep learning",
              "computational biology", "bioinformatics", "genomics"],
    test_suite='tests',
    tests_require=test_requirements
)
