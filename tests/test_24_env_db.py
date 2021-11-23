import json
import os

import kipoi
from kipoi.cli.env import generate_env_db_entry, get_envs_by_model
from kipoi.env_db import EnvDb


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
        for a_el, b_el in zip(a, b):
            assert_rec(a_el, b_el)
    else:
        assert a == b


def test_env_db_kipoi(tmpdir, monkeypatch):
    # Test the kipoi vs. dir path ambiguation
    # Test the DeepSEA model using the `kipoi` and the `dir` sources
    # Test the `shared/envs/kipoi-py3-keras1.2.yaml` model using the `kipoi` and the `dir` sources
    json_file = os.path.join(str(tmpdir), "db.json")
    sample_cli_path = os.path.join(str(tmpdir), "sample")
    with open(sample_cli_path, "w") as fh:
        fh.write("")

    db = EnvDb(json_file)
    kwargs = {"dataloader": [], "gpu": True, "model": None, "source": "kipoi",
              "tmpdir": "something"}

    # generate the kipoi entries
    kipoi_entries = []
    for model in [["DeepSEA"], ["shared/envs/kipoi-py3-keras1.2"]]:
        kwargs['model'] = model
        db_entry = generate_env_db_entry(get_args(kwargs)())
        db.append(db_entry)
        kipoi_entries.append(db_entry)

    # generate the kipoi entries
    dir_entries = []
    local_path = kipoi.get_source("dir").local_path
    kwargs["source"] = "dir"
    for model in [["example/models/pyt"], ["example/models/shared/envs/kipoi-py3-keras1.2"]]:
        kwargs['model'] = [os.path.join(local_path,model[0])]
        db_entry = generate_env_db_entry(get_args(kwargs)())
        db.append(db_entry)
        dir_entries.append(db_entry)

    # make sure there is no mixup between the kipoi and dir models and make sure the full path is only used
    # for dir models

    assert db.get_entry_by_model("DeepSEA", only_most_recent=False) == [kipoi_entries[0]]
    assert db.get_entry_by_model("CpGenie/merged", only_most_recent=False) == [dir_entries[1], kipoi_entries[1]]
    assert db.get_entry_by_model(os.path.join(local_path, "example/models/pyt"),
                                 only_most_recent=False) == [dir_entries[0]]

    # monkeypatch the get_model_env_db()
    monkeypatch.setattr(kipoi.env_db, 'get_model_env_db', lambda: db)

    assert get_envs_by_model(['DeepSEA'], "kipoi", only_most_recent=False, only_valid=False) == [kipoi_entries[0]]
    assert get_envs_by_model(["CpGenie/merged"], "kipoi", only_most_recent=False,
                             only_valid=False) == [dir_entries[1],kipoi_entries[1]]
    assert get_envs_by_model(["example/models/pyt"], "dir", only_most_recent=False,
                             only_valid=False) == [dir_entries[0]]




def test_env_db(tmpdir):
    json_file = os.path.join(str(tmpdir), "db.json")
    sample_cli_path = os.path.join(str(tmpdir), "sample")
    with open(sample_cli_path, "w") as fh:
        fh.write("")

    db = EnvDb(json_file)
    kwargs = {"dataloader": [], "env": "test_env", "gpu": True, "model": None, "source": "dir",
              "tmpdir": "something"}

    entries = []
    source_path = kipoi.get_source("dir").local_path
    for model in [["example/models/pyt"], ["example/models/shared/envs/kipoi-py3-keras1.2", "example/models/pyt"]]:
        kwargs['model'] = model
        db_entry = generate_env_db_entry(get_args(kwargs)())
        db.append(db_entry)
        entries.append(db_entry)

    pyt_query_name = os.path.join(source_path, "example/models/pyt")

    assert db.get_entry_by_model(pyt_query_name) == entries[1]
    assert db.get_entry_by_model(pyt_query_name + "_class") is None
    assert db.get_entry_by_model(pyt_query_name, only_most_recent=False) == entries[::-1]

    # test if the viability check is ok:
    entry = db.get_entry_by_model(pyt_query_name)
    entry.successful = True
    entry.cli_path = sample_cli_path
    assert db.get_entry_by_model(pyt_query_name, only_most_recent=False, only_valid=True) == [entry]
    entry.successful = False
    assert len(db.get_entry_by_model(pyt_query_name, only_most_recent=False, only_valid=True)) == 0
    entry.successful = True
    entry.cli_path = None
    assert len(db.get_entry_by_model(pyt_query_name, only_most_recent=False, only_valid=True)) == 0

    db.save()
    del db

    # Test if loading is fine
    db2 = EnvDb(json_file)
    # test dict identity
    assert_rec(db2.get_entry_by_model(pyt_query_name).get_config(), entries[1].get_config())
    assert db2.get_entry_by_model(pyt_query_name + "_class") is None

    del db2

    # Test if bad entries are skipped
    with open(json_file, "r") as fh:
        db_dict = json.load(fh)

    # Add a bad entry:
    new_key = max([int(k) for k in db_dict["_default"]]) + 1
    db_dict["_default"][str(new_key)] = {"conda_version": "conda 4.5.4", "kipoi_version": "0.5.6"}

    with open(json_file, "w") as fh:
        json.dump(db_dict, fh)

    # Check if there is a warning
    # with pytest.warns(UserWarning): # There seems to be a general problem with warnings...
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
