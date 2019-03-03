from kipoi_utils.external.related.mixins import RelatedConfigMixin
from kipoi_utils.external.related.fields import StrSequenceField
import os
import kipoi
import related
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_MODEL_ENV_DB = None


def get_model_env_db():
    global _MODEL_ENV_DB
    if _MODEL_ENV_DB is None:
        _MODEL_ENV_DB = EnvDb(kipoi.config._env_db_path)
    return _MODEL_ENV_DB


def reload_model_env_db():
    global _MODEL_ENV_DB
    if _MODEL_ENV_DB is not None:
        del _MODEL_ENV_DB
    _MODEL_ENV_DB = None
    get_model_env_db()


@related.mutable
class EnvCreateArgs(RelatedConfigMixin):
    model = StrSequenceField(str, required=True)
    source = related.StringField(required=True)
    dataloader = StrSequenceField(str, default=[], required=False)
    env = related.StringField(default=None, required=False)
    gpu = related.BooleanField(default=False, required=False)
    tmpdir = related.StringField(default=None, required=False)
    vep = related.BooleanField(default=False, required=False)


@related.mutable
class EnvDbEntry(RelatedConfigMixin):
    """
    Compatible Models explicitly and the possible (intermediate) groups (derive from kipoi/models/shared/envs/models.yaml (can be model group too) or models in the group if it was a group: sub_models = list_subcomponents(parsed_model, parsed_source, "model") - TODO - check kipoi.cli.env.merge_deps)
    Store all the parsed args
    Store the conda version
    Store the kipoi version
    Timestamp
    successful
    cli_path
    """
    conda_version = related.StringField(required=True)
    kipoi_version = related.StringField(required=True)
    timestamp = related.FloatField(required=True)
    compatible_models = related.SequenceField(str, required=True)
    create_args = related.ChildField(EnvCreateArgs,
                                     required=True)
    successful = related.BooleanField(default=False, required=False)
    cli_path = related.StringField(default=None, required=False)


class EnvDb:
    def __init__(self, db_path):
        from tinydb import TinyDB
        self.entries = []
        self.db = TinyDB(db_path)
        # There is a main problem with querying the way it is done here. The content of the DB is only loaded once
        # and therefore the DB is not used as a DB and the object must the re-instantiated to update the entries.
        self._load_entries()

    def _load_entries(self):
        self.entries = []
        # load everything into memory
        for db_entry in self.db:
            try:
                self.entries.append(EnvDbEntry.from_config(db_entry))
            except Exception as e:
                logger.warning("Could not load entry with cli path {0} due to: {1}. "
                            "Skipping...".format(str(db_entry), str(e)))

    def get_entry_by_model(self, model_name, only_most_recent=True, only_valid=False):
        # iterate over all the entries and select the ones where the model_name is part of one of the listed models
        # For checking split the model_name by "/" as well as the env-compatible model names and then check equality.
        # Select the one with the most recent timestamp
        norm_name = lambda x: x.lstrip("/").rstrip("/")
        norm_model_name = norm_name(model_name)
        query_model_tk_len = len(norm_model_name.split("/"))
        sel_entries = {}
        for entry in self.get_all(only_valid=only_valid):
            pre_sel = [m for m in entry.compatible_models if model_name in m]
            sel = [m for m in pre_sel if "/".join(norm_name(m).split("/")[:query_model_tk_len]) == norm_model_name]
            if len(sel) != 0:
                sel_entries[entry.timestamp] = entry
        ordered_entries = OrderedDict([(k, sel_entries[k]) for k in sorted(list(sel_entries.keys()))][::-1])

        if only_most_recent:
            if len(ordered_entries) == 0:
                return None
            else:
                return list(ordered_entries.values())[0]
        else:
            return list(ordered_entries.values())

    def get_all_unfinished(self):
        unfinished = []
        for e in self.entries:
            if not e.successful or e.cli_path is None or not os.path.exists(e.cli_path):
                unfinished.append(e)
        return unfinished

    def db_remove_unfinished(self):
        [self.remove(e) for e in self.get_all_unfinished()]

    def get_all(self, only_valid=False):
        entries = self.entries
        if only_valid:
            invalid = self.get_all_unfinished()
            entries = [e for e in entries if e not in invalid]
        return entries

    def remove(self, entry):
        self.entries = [e for e in self.entries if e != entry]

    def append(self, entry):
        self.entries.append(entry)

    def save(self):
        self.db.purge()
        for entry in self.entries:
            self.db.insert(entry.get_config())

    def __del__(self):
        self.db.close()
