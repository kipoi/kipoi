import pytest
from related import from_yaml

from kipoi.components import *
from kipoi.components import VarEffectFuncType

yaml_in_no_args = """type: variant_effects
"""
def test_insufficient_info():
    with pytest.raises(Exception):
        # args.seq_input is required.
        pps = PostProcModelStruct.from_config(from_yaml(yaml_in_no_args))
    #
    pps = PostProcDataloaderStruct.from_config(from_yaml(yaml_in_no_args))
    assert pps.args is None


yaml_in_simple = """type: variant_effects
args:
  seq_input:
    - seq
"""
def test_minimal_info():
    pps = PostProcModelStruct.from_config(from_yaml(yaml_in_simple))
    assert pps.type is PostProcType.VAR_EFFECT_PREDICTION
    assert pps.args.seq_input == ["seq"] # should always be there and is always a list of strings
    assert pps.args.use_rc == VarEffectRCTypes.none



yaml_in_simple_rc = """type: variant_effects
args:
  seq_input:
    - seq
  use_rc: seq_only
"""
def test_use_rc():
    pps = PostProcModelStruct.from_config(from_yaml(yaml_in_simple_rc))
    assert pps.type is PostProcType.VAR_EFFECT_PREDICTION
    assert pps.args.seq_input == ["seq"] # should always be there and is always a list of strings
    assert pps.args.use_rc == VarEffectRCTypes.seq_only


yaml_in_bed = """type: variant_effects
args:
  bed_input:
    - intervals_file
"""
def test_dataloader_bed_input():
    pps = PostProcDataloaderStruct.from_config(from_yaml(yaml_in_bed))
    assert pps.type is PostProcType.VAR_EFFECT_PREDICTION
    assert pps.args.bed_input == ["intervals_file"] # pps.args may be None



yaml_in = """type: variant_effects
args:
  seq_input:
    - seq
  scoring_functions:
    - name: diff
      type: diff
    - type: logit
    - type: deepsea_scr
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
    assert pps.type is PostProcType.VAR_EFFECT_PREDICTION
    assert pps.args.seq_input == ["seq"] # should always be there and is always a list of strings
    scoring_fns = [{"name": "diff", "type": VarEffectFuncType.diff, "default": False},
                   {"type": VarEffectFuncType.logit, "default": False},
                   {"default": True, "type": VarEffectFuncType.deepsea_scr},
                   {"name": "mydiff", "type": VarEffectFuncType.custom, "defined_as": "postproc.py::myfun", "default": False}]

    for in_obj, fn in zip(pps.args.scoring_functions, scoring_fns):
        for k in fn:
            if k == "type":
                assert in_obj.type is fn["type"]
            else:
                assert getattr(in_obj, k) == fn[k]

    expected_args = {"first_arg": {"doc": "blablabla1", "default": "1"}, "second_arg": {"doc": "blablabla", "default": "10"}}
    custom_fn_args = pps.args.scoring_functions[-1].args
    for k in expected_args:
        for k2 in expected_args[k]:
            assert getattr(custom_fn_args[k],k2) == expected_args[k][k2]

