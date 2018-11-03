import pytest
from kipoi.cli.env_db import EnvDb
from kipoi.cli.env import generate_env_db_entry
import os
import json
import warnings


def get_args(def_kwargs):
    class dummy_args:
        kwargs = def_kwargs
        model = kwargs["model"]
        source = kwargs["source"]

        def _get_kwargs(self):
            return self.kwargs

    return dummy_args

def assert_rec(a, b):
    if isinstance(a, dict):
        assert set(a.keys()) == set(b.keys())
        for k in a:
            assert_rec(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for a_el, b_el in zip(a,b):
            assert_rec(a_el, b_el)
    else:
        assert a == b

def test_env_db(tmpdir):
    json_file = os.path.join(tmpdir, "db.json")
    db = EnvDb(json_file)
    kwargs = {"dataloader": [], "env": "test_env", "gpu": True, "model": None, "source": "dir",
              "tmpdir": "something", "vep": True}

    entries = []
    for model in [["example/models/pyt"], ["example/models/shared/envs/kipoi-py3-keras1.2", "example/models/pyt"]]:
        kwargs['model'] = model
        db_entry = generate_env_db_entry(get_args(kwargs)())
        db.append(db_entry)
        entries.append(db_entry)

    assert db.get_entry_by_model("example/models/pyt") == entries[1]
    assert db.get_entry_by_model("example/models/pyt_class") is None
    assert db.get_entry_by_model("example/models/pyt", only_most_recent = False) == entries[::-1]

    db.save()
    del db

    # Test if loading is fine
    db2 = EnvDb(json_file)
    # test dict identity
    assert_rec(db2.get_entry_by_model("example/models/pyt").get_config(), entries[1].get_config())
    assert db2.get_entry_by_model("example/models/pyt_class") is None

    del db2

    # Test if bad entries are skipped
    with open(json_file, "r") as fh:
        db_dict = json.load(fh)

    # Add a bad entry:
    new_key = max([int(k) for k in db_dict["_default"]])+1
    db_dict["_default"][str(new_key)] = {"conda_version": "conda 4.5.4", "kipoi_version": "0.5.6"}

    with open(json_file, "w") as fh:
        json.dump(db_dict, fh)

    # Check if there is a warning
    #with pytest.warns(UserWarning): # There seems to be a general problem with warnings...
    db_warns = EnvDb(json_file)

    assert len(db_warns.entries) == 2

    # Now save so that the bad entry is be gone
    db_warns.save()
    del db_warns

    # Make sure the bad entry is not there anymore
    with open(json_file, "r") as fh:
        db_dict_recovered = json.load(fh)

    found = 0
    for val in db_dict_recovered['_default'].values():
        found += int(val == db_dict["_default"][str(new_key)])

    assert len(db_dict_recovered["_default"]) == new_key - 1
    assert found == 0

    os.unlink(json_file)

