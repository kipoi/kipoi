from __future__ import absolute_import
from __future__ import print_function

import abc
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class BaseModel(object):
    __metaclass__ = abc.ABCMeta

    MODEL_PACKAGE = None

    @abc.abstractmethod
    def predict_on_batch(self, x):
        raise NotImplementedError

    @classmethod
    def _sufficient_deps(cls, deps):
        """Tests it the provided dependencies contain MODEL_PACKAGE

        Args:
          deps: instance of kipoi.specs.Dependencies

        Returns:
          True if cls.MODEL_PACKAGE is listed in the depenencies and False otherwise
        """
        if cls.MODEL_PACKAGE is None:
            return True
        else:
            for d in deps.conda:
                if cls.MODEL_PACKAGE in d:
                    return True
            for d in deps.pip:
                if cls.MODEL_PACKAGE in d:
                    return True
            return False


    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass