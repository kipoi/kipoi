"""Test $ kipoi env ... functions
"""
import subprocess
import os
from kipoi.utils import read_yaml
from kipoi.cli.env import get_env_name, export_env, merge_deps
from kipoi.remote import list_subcomponents


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
    env, env_file = export_env(["rbp_eclip/UPF1"],
                               env_dir=str(tmpdir))
    env_dict = read_yaml(env_file)
    assert env_dict['channels'] == ['bioconda', 'conda-forge', 'defaults']

    pip_idx = [i for i, x in enumerate(env_dict['dependencies']) if isinstance(x, dict)][0]
    assert [p for p in env_dict['dependencies'][pip_idx]['pip']
            if "tensorflow=" in p or "tensorflow>=" in p]

    for dep in ['concise']:
        assert [p for p in env_dict['dependencies'][pip_idx]['pip']
                if dep in p]

    # gpu version
    env, env_file = export_env(["rbp_eclip/UPF1"],
                               env_dir=str(tmpdir),
                               gpu=True
                               )
    env_dict = read_yaml(env_file)
    assert env_dict['channels'] == ['bioconda', 'conda-forge', 'defaults']
    assert [p for p in env_dict['dependencies'][pip_idx]['pip'] if "tensorflow-gpu=" in p or "tensorflow-gpu>=" in p]

    # vep
    env, env_file = export_env(["rbp_eclip/UPF1"],
                               env_dir=str(tmpdir),
                               gpu=True,
                               vep=True,
                               )
    env_dict = read_yaml(env_file)
    assert env_dict['channels'] == ['bioconda', 'conda-forge', 'defaults']
    assert [p for p in env_dict['dependencies'] if "cyvcf2" in p]
    assert env_dict['name'] == env


def test_cli(tmpdir):
    env_file = os.path.join(str(tmpdir), "env.yml")
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "env", "export",
            "rbp_eclip/UPF1",
            "-o", env_file]
    returncode = subprocess.call(args=args)
    assert returncode == 0
    env_dict = read_yaml(env_file)
    env, env_file2 = export_env(["rbp_eclip/UPF1"],
                                env_dir=str(tmpdir))
    env_dict2 = read_yaml(env_file2)
    assert env_dict == env_dict2


def test_export_multiple(tmpdir):
    """Test the export functionality
    """

    tmpdir = "/tmp/test/"
    # makefile
    env, env_file = export_env(["rbp_eclip/UPF1", "rbp_eclip/XRN2"],
                               env_dir=str(tmpdir))
    env_dict = read_yaml(env_file)
    del env_dict['name']
    env, env_file = export_env(["rbp_eclip/UPF1"],
                               env_dir=str(tmpdir))
    env_dict2 = read_yaml(env_file)
    del env_dict2['name']

    assert env_dict == env_dict2

    env, env_file = export_env(["rbp_eclip/UPF1", "HAL"],
                               env_dir=str(tmpdir))
    env_dict2 = read_yaml(env_file)
    del env_dict2['name']

    assert env_dict != env_dict2


def test_list_submodules():
    assert set(list_subcomponents("MaxEntScan", "kipoi", "model")) == {"MaxEntScan/3prime", "MaxEntScan/5prime"}
    assert set(list_subcomponents("Basenji", "kipoi", "model")) == {"Basenji"}


def test_deps():
    assert merge_deps(["MaxEntScan"]) == merge_deps(["MaxEntScan/5prime"])
