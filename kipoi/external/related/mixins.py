from __future__ import absolute_import
from __future__ import print_function

import logging

import os
import related
from attr._make import fields

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class RelatedConfigMixin(object):
    """Provides from_config and get_config to @related.immutable decorated classes
    """
    @classmethod
    def from_config(cls, cfg):
        # TODO - create a nicer error message - see above
                # Verbose unrecognized field
        # for k in kwargs.keys():
        #     if k not in cls.REQ_FIELDS + cls.OPT_FIELDS:
        #         raise ValueError("Unrecognized field in info: '{f}'. Avaiable fields are: {l}".
        #                          format(f=k, l=cls.REQ_FIELDS))

        # # Verbose undefined field
        # undefined_set = set(cls.REQ_FIELDS) - kwargs.keys()
        # if undefined_set:
        #     raise ValueError("The following arguments were not specified: {0}. Please specify them.".
        #                      format(undefined_set))
        attrs = fields(cls)
        cls_keys = {a.metadata.get('key') or a.name for a in attrs}
        cfg_keys = set(cfg.keys())
        extra_keys = cfg_keys - cls_keys
        if len(extra_keys) > 0:
            raise ValueError("Unrecognized fields: {0}. Available fields are {1}".format(extra_keys, cls_keys))

        return related.to_model(cls, cfg)

    def get_config(self):
        return related.to_dict(self)

    def get_config_as_yaml(self):
        generated_yaml = related.to_yaml(self,
                                         suppress_empty_values=True,
                                         suppress_map_key_values=True)
        return generated_yaml


class RelatedLoadSaveMixin(RelatedConfigMixin):
    """Adds load and dump on top of RelatedConfigMixin for reading and writing from a yaml file
    """

    @classmethod
    def load(cls, path, append_path=True):
        """Loads model from a yaml file
        """
        original_yaml = open(path).read().strip()
        parsed_dict = related.from_yaml(original_yaml)
        if append_path and "path" not in parsed_dict:
            parsed_dict["path"] = path
        try:
            return cls.from_config(parsed_dict)
        except Exception as e:
            raise Exception("Unable to load file {0} into class {1}.\nError: \n{2}".
                            format(os.path.abspath(path), cls, str(e)))

    @classmethod
    def from_string(cls, string):
        """Loads model from a yaml file
        """
        parsed_dict = related.from_yaml(string)
        try:
            return cls.from_config(parsed_dict)
        except Exception as e:
            raise Exception("Unable to load yaml string: {0} into class {1}.\nError: \n{2}".
                            format(os.path.abspath(string), cls, str(e)))

    def dump(self, path):
        """Dump the object to a yaml file
        """
        generated_yaml = related.to_yaml(self,
                                         suppress_empty_values=True,
                                         suppress_map_key_values=True)  # .strip()
        with open(path, "w") as f:
            f.write(generated_yaml)
