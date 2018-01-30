from __future__ import absolute_import
from __future__ import print_function
import sys


class bcolors:
    """https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


model_type = '{{ cookiecutter.model_type }}'
model_input_type = '{{ cookiecutter.model_input_type }}'
model_output_type = '{{ cookiecutter.model_output_type }}'

if model_type == "sklearn":
    if model_input_type is not "np.array" or model_output_type is not "np.array":
        print(bcolors.FAIL + "\nERROR: " + bcolors.ENDC + "model_input_type and model_output_type need to be 'np.array' for model_type == 'sklearn'")
        # exits with status 1 to indicate failure
        sys.exit(1)
