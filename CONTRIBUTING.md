# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/kipoi/kipoi/issues>.

If you are reporting a bug, please include:

-   Your operating system name and version.
-   Any details about your local setup that might be helpful in troubleshooting.
-   Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with “bug” and “help wanted” is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with “enhancement” and “help wanted” is open to whoever wants to implement it.

### Write Documentation

Kipoi could always use more documentation, whether as part of the official Kipoi docs, in docstrings, or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at <https://github.com/kipoi/kipoi/issues>.

If you are proposing a feature:

-   Explain in detail how it would work.
-   Keep the scope as narrow as possible, to make it easier to implement.
-   Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Workflow

- make an issue for the thing you want to implement
- create the corresponding branch
- develop
- write units tests in tests/
- write documentation in markdown (see other functions for example)
- push the changes
- make a pull request
- once the pull request is merged, the issue will be closed

## Get Started!

Ready to contribute? Here’s how to set up kipoi for local development.

1.  Fork the kipoi repo on GitHub.
2.  Clone your fork locally:

        $ git clone git@github.com:your_name_here/kipoi.git

3.  Install your local copy into a conda environment. Assuming you have conda installed, this is how you set up your fork for local development:

        $ conda create -n kipoi-py35 python=3.5
        $ cd kipoi/
		$ source activate kipoi-py35
        $ pip install -e '.[develop]'

4.  Create a branch for local development:

        $ git checkout -b name-of-your-bugfix-or-feature

    Now you can make your changes locally.

5.  When you’re done making changes, check that your changes pass the tests:

        $ py.test tests/ -n 4

Where `-n 4` will use 4 cores in parallel to run tests.

6.  Commit your changes and push your branch to GitHub:

        $ git add .
        $ git commit -m "Your detailed description of your changes."
        $ git push origin name-of-your-bugfix-or-feature

7.  Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1.  The pull request should include tests.
2.  If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring.
3.  The pull request should work for Python 2.7, 3.5 and 3.6.
