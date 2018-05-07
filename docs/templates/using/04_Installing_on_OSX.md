## Using Kipoi - Installing on OSX

Depending on the versino of OSX you are using there is python pre-installed or not. On OSX Sierra it is not, but on OSX High Sierra it is.
For Kipoi to work fully you will need a version of python (2.7, 3.5 or 3.6) installed, preferably you will also have an installation of conda.
We have seen problems when conda environments were re-used so we strongly recommend that you create a new environment e.g. `kipoi` where you
install Kipoi.

### Steps

#### Make sure you have python installed:

You can try by just execting `python` in your Terminal, if nothing is found you will want to install python (not `pythonw`).
There are some good explanations on how [python 2 can be installed on OSX Sierra](http://docs.python-guide.org/en/latest/starting/install/osx/)
and if you are using High Sierra and you prefer python 3 you can [follow this](http://docs.python-guide.org/en/latest/starting/install3/osx/).

After completing the steps and installing [conda or miniconda](https://conda.io/) please procede as described in [getting started](./01_Getting_started).
