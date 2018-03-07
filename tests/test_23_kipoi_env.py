"""Test $ kipoi env ... functions
"""
from kipoi.utils import read_yaml
from kipoi.cli.env import get_env_name, export_env


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
    assert env_dict['channels'] == ['defaults']
    assert [p for p in env_dict['dependencies'][1]['pip'] if "tensorflow=" in p]

    # gpu version
    env, env_file = export_env(["rbp_eclip/UPF1"],
                               env_dir=str(tmpdir),
                               gpu=True
                               )
    env_dict = read_yaml(env_file)
    assert env_dict['channels'] == ['defaults']
    assert [p for p in env_dict['dependencies'][1]['pip'] if "tensorflow-gpu=" in p]

    # vep
    env, env_file = export_env(["rbp_eclip/UPF1"],
                               env_dir=str(tmpdir),
                               gpu=True,
                               vep=True,
                               )
    env_dict = read_yaml(env_file)
    assert env_dict['channels'] == ['defaults', 'bioconda']
    assert [p for p in env_dict['dependencies'] if "cyvcf2" in p]
    assert env_dict['name'] == env


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
