from related import from_yaml

from kipoi.components import PostProcModelStruct, PostProcDataLoaderStruct
from kipoi.postprocessing.variant_effects.components import VarEffectFuncType

yaml_in_no_args = """
variant_effects:
"""


def test_insufficient_info():
    assert PostProcModelStruct.from_config(from_yaml(yaml_in_no_args)).variant_effects is None


yaml_in_simple = """
variant_effects:
  seq_input:
    - seq
"""


def test_minimal_info():
    pps = PostProcModelStruct.from_config(from_yaml(yaml_in_simple))
    assert pps.variant_effects is not None
    assert pps.variant_effects.seq_input == ["seq"]  # should always be there and is always a list of strings
    assert not pps.variant_effects.use_rc


yaml_in_simple_rc = """
variant_effects:
  seq_input:
    - seq
  use_rc: True
"""


def test_use_rc():
    pps = PostProcModelStruct.from_config(from_yaml(yaml_in_simple_rc))
    assert pps.variant_effects is not None
    assert pps.variant_effects.seq_input == ["seq"]  # should always be there and is always a list of strings
    assert pps.variant_effects.use_rc


yaml_in_bed = """
variant_effects:
  bed_input:
    - intervals_file
"""


def test_dataloader_bed_input():
    pps = PostProcDataLoaderStruct.from_config(from_yaml(yaml_in_bed))
    assert pps.variant_effects is not None
    assert pps.variant_effects.bed_input == ["intervals_file"]  # pps.args may be None


yaml_in = """
variant_effects:
  seq_input:
    - seq
  scoring_functions:
    - name: diff
      type: diff
    - type: logit
    - type: deepsea_effect
      default: True
    - name: mydiff
      type: custom
      defined_as: postproc.py::myfun
      args:
        first_arg:
          doc: blablabla1
          default: 1
        second_arg:
          doc: blablabla
          default: 10
"""


def test_complex_example():
    pps = PostProcModelStruct.from_config(from_yaml(yaml_in))
    ppsv = pps.variant_effects
    assert ppsv.seq_input == ["seq"]  # should always be there and is always a list of strings
    scoring_fns = [{"name": "diff", "type": VarEffectFuncType.diff, "default": False},
                   {"type": VarEffectFuncType.logit, "default": False},
                   {"default": True, "type": VarEffectFuncType.deepsea_effect},
                   {"name": "mydiff", "type": VarEffectFuncType.custom, "defined_as": "postproc.py::myfun", "default": False}]

    for in_obj, fn in zip(ppsv.scoring_functions, scoring_fns):
        for k in fn:
            if k == "type":
                assert in_obj.type is fn["type"]
            else:
                assert getattr(in_obj, k) == fn[k]

    expected_args = {"first_arg": {"doc": "blablabla1", "default": "1"}, "second_arg": {"doc": "blablabla", "default": "10"}}
    custom_fn_args = ppsv.scoring_functions[-1].args
    for k in expected_args:
        for k2 in expected_args[k]:
            assert getattr(custom_fn_args[k], k2) == expected_args[k][k2]