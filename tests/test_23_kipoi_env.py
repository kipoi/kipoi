"""Test $ kipoi env ... functions
"""
import subprocess
import os
from kipoi_utils.utils import read_yaml
from kipoi.cli.env import get_env_name, export_env, merge_deps, split_models_special_envs, generate_env_db_entry
from kipoi.sources import list_subcomponents
from kipoi.specs import Dependencies
from utils import cp_tmpdir
import pytest


def test_env_name():
    assert get_env_name("mm") == "kipoi-mm"
    assert get_env_name("mm/dd") == "kipoi-mm__dd"
    assert get_env_name("mm", gpu=True) == "kipoi-gpu-mm"
    assert get_env_name(["mm"]) == "kipoi-mm"
    assert get_env_name(["mm"], ["mm"]) == "kipoi-mm"
    assert get_env_name(["mm"], ["mm"]) == "kipoi-mm"
    assert get_env_name(["mm"], "mm") == "kipoi-mm"
    assert get_env_name("mm", "mm") == "kipoi-mm"
    assert get_env_name("mm", []) == "kipoi-mm"

    assert get_env_name(["mm", "mm2"]) == "kipoi-mm,mm2"
    assert get_env_name(["mm", "mm2"], source="source") == "source-mm,mm2"
    assert get_env_name(["foo::mm", "bar::mm2"], source="source") == "source-foo::mm,bar::mm2"
    assert get_env_name("mm", "dl") == "kipoi-mm-DL-dl"
    assert get_env_name("mm/1", "dl/2") == "kipoi-mm__1-DL-dl__2"
    assert get_env_name(["mm"], ["dl"]) == "kipoi-mm-DL-dl"
    assert get_env_name(["mm"], ["dl"]) == "kipoi-mm-DL-dl"
    assert get_env_name(["mm"], ["dl", "dl2"]) == "kipoi-mm-DL-dl,dl2"
    assert get_env_name(["mm"], ["dl", "dl2"], gpu=True) == "kipoi-gpu-mm-DL-dl,dl2"


def test_export(tmpdir):
    """Test the export functionality
    """

    tmpdir = "/tmp/test/"
    # makefile
    env, env_file = export_env(["rbp_eclip/XRCC6"],
                               env_dir=str(tmpdir))
    env_dict = read_yaml(env_file)
    assert sorted(env_dict['channels']) == ['bioconda', 'conda-forge', 'defaults']

    pip_idx = [i for i, x in enumerate(env_dict['dependencies']) if isinstance(x, dict)][0]
    assert [p for p in env_dict['dependencies'][pip_idx]['pip']
            if "tensorflow=" in p or "tensorflow>=" in p]

    for dep in ['concise']:
        assert [p for p in env_dict['dependencies'][pip_idx]['pip']
                if dep in p]

    # gpu version
    env, env_file = export_env(["rbp_eclip/XRCC6"],
                               env_dir=str(tmpdir),
                               gpu=True
                               )
    env_dict = read_yaml(env_file)
    assert sorted(env_dict['channels']) == ['bioconda', 'conda-forge', 'defaults']
    assert [p for p in env_dict['dependencies'][pip_idx]['pip'] if "tensorflow-gpu=" in p or "tensorflow-gpu>=" in p]



def test_cli(tmpdir):
    env_file = os.path.join(str(tmpdir), "env.yml")
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "env", "export",
            "rbp_eclip/XRCC6",
            "-o", env_file]
    returncode = subprocess.call(args=args)
    assert returncode == 0
    env_dict = read_yaml(env_file)
    env, env_file2 = export_env(["rbp_eclip/XRCC6"],
                                env_dir=str(tmpdir))
    env_dict2 = read_yaml(env_file2)
    assert env_dict == env_dict2


def test_export_multiple(tmpdir):
    """Test the export functionality
    """

    tmpdir = "/tmp/test/"
    # makefile
    env, env_file = export_env(["rbp_eclip/XRCC6", "rbp_eclip/TIA1"],
                               env_dir=str(tmpdir))
    env_dict = read_yaml(env_file)
    del env_dict['name']
    env, env_file = export_env(["rbp_eclip/XRCC6"],
                               env_dir=str(tmpdir))
    env_dict2 = read_yaml(env_file)
    del env_dict2['name']

    assert env_dict == env_dict2

    env, env_file = export_env(["rbp_eclip/XRCC6", "HAL"],
                               env_dir=str(tmpdir))
    env_dict2 = read_yaml(env_file)
    del env_dict2['name']

    assert env_dict != env_dict2


def test_list_submodules():
    assert set(list_subcomponents("MaxEntScan", "kipoi", "model")) == {"MaxEntScan/3prime", "MaxEntScan/5prime"}
    assert set(list_subcomponents("Basenji", "kipoi", "model")) == {"Basenji"}


def test_deps():
    assert merge_deps(["MaxEntScan"]) == merge_deps(["MaxEntScan/5prime"])

    # test mix of special environments and models
    merge_deps(["example/models/shared/envs/kipoi-py3-keras1.2"], source="dir")
    with pytest.raises(ValueError):
        merge_deps(["example/models/shared/envs/kipoi-py3-keras1.2.yaml"], source="dir")

    with pytest.raises(Exception):
        merge_deps(["example/models/shared/envs/kipoi-py3-keras1.2_bad"], source="dir")

    deps = merge_deps(["example/models/shared/envs/kipoi-py3-keras1.2", "example/models/pyt"], source="dir")
    assert len([el for el in deps.conda if "pytorch" in el]) == 1


def get_args(def_kwargs):
    class dummy_args:
        kwargs = def_kwargs
        model = kwargs["model"]
        source = kwargs["source"]

        def _get_kwargs(self):
            return self.kwargs

    return dummy_args


def test_generate_env_db_entry():
    # test in general and test whether the automatic generation of sub-models works, also in combination
    # with a clearly defined model
    import yaml
    import kipoi
    import time
    from kipoi.cli.parser_utils import parse_source_name
    kwargs = {"dataloader": [], "env": "test_env", "gpu": True, "model": None, "source": "dir",
              "tmpdir": "something"}
    source_path = kipoi.get_source("dir").local_path
    kipoi_path = kipoi.get_source("kipoi").local_path
    for model in [["example/models/pyt"], ["example/models/shared/envs/kipoi-py3-keras1.2", "example/models/pyt"]]:
        kwargs['model'] = model
        db_entry = generate_env_db_entry(get_args(kwargs)())
        assert all([kwargs[k] == getattr(db_entry.create_args, k) for k in kwargs])

        # generate the reference output
        special_envs, only_models = split_models_special_envs(model)
        sub_models = []
        for model in only_models:
            parsed_source, parsed_model = parse_source_name(kwargs["source"], model)
            sub_models.extend([os.path.join(source_path, e) for e in
                               list_subcomponents(parsed_model, parsed_source, "model")])
        if len(special_envs) != 0:
            with open("example/models/shared/envs/models.yaml", "r") as fh:
                special_env_models = yaml.safe_load(fh)
            for special_env in special_envs:
                for model_group_name in special_env_models[os.path.basename(special_env)]:
                    sub_models.extend([e for e in
                                       list_subcomponents(model_group_name, "kipoi", "model")])

        assert set(db_entry.compatible_models) == set(sub_models)
        assert db_entry.cli_path is None
        assert db_entry.successful == False
        assert db_entry.kipoi_version == kipoi.__version__
        assert db_entry.timestamp < time.time()


def test_split_models_special_envs():
    # simple test if splitting works:
    special = ["example/models/shared/envs/py3.5-keras1.2"]
    conventional = ["example/models/pyt"]
    special_envs, only_models = split_models_special_envs(special + conventional)
    assert special_envs == special
    assert only_models == conventional


def test_decorator_env_loading(tmpdir):
    mdir = cp_tmpdir("example/models/kipoi_dataloader_decorator", tmpdir)
    assert merge_deps([mdir], source='dir') == \
           Dependencies(conda=['python=2.7', 'scikit-learn'],
                        pip=['kipoi', 'scikit-learn', 'tqdm'],
                        conda_channels=['defaults'])
