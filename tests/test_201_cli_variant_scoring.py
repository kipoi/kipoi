from kipoi.components import PostProcModelStruct
from related import from_yaml
import pytest
from kipoi.cli.postproc import _get_avail_scoring_methods, _get_scoring_fns, builtin_default_kwargs
from kipoi.postprocessing import variant_effects as ve


class dummy_container(object):
    pass


postproc_yaml = """
variant_effects:
  seq_input:
    - seq
  scoring_functions:%s
    - type: logit
    - type: deepsea_scr
      default: True
    - name: mydiff
      type: custom
      defined_as: tests/data/dummy_diff.py::LogitAlt
%s
"""


diff_str = """
    - name: diff
      type: diff
"""

args_w_default = """      args:
        rc_merging:
          default: "max"
"""

optional_args = """      args:
        rc_merging:
          optional: true
"""


def test_custom_fns():
    template_avail_scoring_fns = [ve.Logit, ve.DeepSEA_effect, ve.LogitAlt]
    template_avail_scoring_fn_labels = ["logit", "deepsea_scr", "mydiff"]

    exp_avail_scoring_fns = [template_avail_scoring_fns + [ve.Diff], [ve.Diff] + template_avail_scoring_fns]
    exp_avail_scoring_fn_labels = [template_avail_scoring_fn_labels + ["diff"], ["diff"] + template_avail_scoring_fn_labels]

    for i, diff_str_here in enumerate(["", diff_str]):
        if diff_str_here == "":
            exp_avail_scoring_fn_def_args = [None, [builtin_default_kwargs] * 2 +
                                             [{"rc_merging": "max"}] + [builtin_default_kwargs],
                                             [builtin_default_kwargs] * 2 + [{}] + [builtin_default_kwargs]]
        else:
            exp_avail_scoring_fn_def_args = [None, [builtin_default_kwargs] * 3 +
                                             [{"rc_merging": "max"}], [builtin_default_kwargs] * 3 + [{}]]
        for i2, mydiff_args in enumerate(["", args_w_default, optional_args]):
            pps = PostProcModelStruct.from_config(from_yaml(postproc_yaml % (diff_str_here, mydiff_args)))
            model = dummy_container()
            model.postprocessing = pps
            if i2 == 0:
                # mydiff has one argument but none are defined.
                with pytest.raises(ValueError):
                    _get_avail_scoring_methods(model)
            else:
                avail_scoring_fns, avail_scoring_fn_def_args, avail_scoring_fn_names, default_scoring_fns =\
                    _get_avail_scoring_methods(model)
                assert all([el1 is el2 for el1, el2 in zip(exp_avail_scoring_fns[i], avail_scoring_fns)])
                assert exp_avail_scoring_fn_labels[i] == avail_scoring_fn_names
                assert all([el1 == el2 for el1, el2 in zip(exp_avail_scoring_fn_def_args[i2], avail_scoring_fn_def_args)])
                assert default_scoring_fns == ["deepsea_scr"]


postproc_yaml_nofndef = """
variant_effects:
  seq_input:
    - seq
"""


# by default at least and only offer the diff functionality
def test_default_diff():
    pps = PostProcModelStruct.from_config(from_yaml(postproc_yaml_nofndef))
    model = dummy_container()
    model.postprocessing = pps
    avail_scoring_fns, avail_scoring_fn_def_args, avail_scoring_fn_names, default_scoring_fns =\
        _get_avail_scoring_methods(model)
    assert all([el1 is el2 for el1, el2 in zip([ve.Diff], avail_scoring_fns)])
    assert all([el1 is el2 for el1, el2 in zip(["diff"], avail_scoring_fn_names)])
    assert all([el1 == el2 for el1, el2 in zip([builtin_default_kwargs], avail_scoring_fn_def_args)])
    assert default_scoring_fns == ["diff"]


# test duplication of names
dupl_name_yaml = """
variant_effects:
  seq_input:
    - seq
  scoring_functions:
    - type: logit
      name: logit
    - type: logit
      name: logit
"""


def test_dupl_name():
    pps = PostProcModelStruct.from_config(from_yaml(dupl_name_yaml))
    model = dummy_container()
    model.postprocessing = pps
    with pytest.raises(Exception):
        _get_avail_scoring_methods(model)


# test modification of name with custom_
rename_custom_yaml = """
variant_effects:
  seq_input:
    - seq
  scoring_functions:
    - name: logit
      type: custom
      defined_as: tests/data/dummy_diff.py::LogitAlt
      args:
        rc_merging:
          default: "max"
"""


def test_rename_custom():
    pps = PostProcModelStruct.from_config(from_yaml(rename_custom_yaml))
    model = dummy_container()
    model.postprocessing = pps
    avail_scoring_fns, avail_scoring_fn_def_args, avail_scoring_fn_names, default_scoring_fns =\
        _get_avail_scoring_methods(model)
    assert avail_scoring_fn_names == ["custom_logit", "diff"]
    assert default_scoring_fns == ["custom_logit"]


postproc_autodefault_yaml = """
variant_effects:
  seq_input:
    - seq
  scoring_functions:
    - type: logit
    - type: deepsea_scr
    - name: mydiff
      type: custom
      defined_as: tests/data/dummy_diff.py::LogitAlt
      args:
        rc_merging:
          default: "max"
"""


# if no default is set all scoring functions are used.
def test_auto_default():
    pps = PostProcModelStruct.from_config(from_yaml(postproc_autodefault_yaml))
    model = dummy_container()
    model.postprocessing = pps
    avail_scoring_fns, avail_scoring_fn_def_args, avail_scoring_fn_names, default_scoring_fns = \
        _get_avail_scoring_methods(model)
    assert default_scoring_fns + ["diff"] == avail_scoring_fn_names


postproc_cli_yaml = """
variant_effects:
  seq_input:
    - seq
  scoring_functions:
    - type: logit
    - type: deepsea_scr
"""


def test__get_scoring_fns():
    pps = PostProcModelStruct.from_config(from_yaml(postproc_cli_yaml))
    model = dummy_container()
    model.postprocessing = pps
    scorers = [{"logit": ve.Logit, "deepsea_scr": ve.DeepSEA_effect}, {"logit": ve.Logit}, {}]
    json_kwargs = "{rc_merging: 'max'}"
    for sel_scoring_labels, scorer in zip([[], ["logit"], ["inexistent"]], scorers):
        jk_list = [json_kwargs] * 2
        if len(sel_scoring_labels) != 0:
            jk_list = [json_kwargs]
        for sel_scoring_kwargs in [[], jk_list]:
            if sel_scoring_labels == ["inexistent"]:
                with pytest.raises(ValueError):
                    _get_scoring_fns(model, sel_scoring_labels, sel_scoring_kwargs)
            else:
                dts = _get_scoring_fns(model, sel_scoring_labels, sel_scoring_kwargs)
                for k in scorer:
                    assert isinstance(dts[k], scorer[k])
