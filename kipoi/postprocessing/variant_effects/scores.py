from __future__ import absolute_import
from __future__ import print_function

import copy
import warnings
import numpy as np
import abc

import kipoi
import logging
import six
from kipoi.postprocessing.variant_effects.components import VarEffectFuncType
from kipoi.utils import load_module, getargs, parse_json_file_str

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def default_kwargs(args):
    """Return the example kwargs
    """
    return {k: v.default for k, v in six.iteritems(args) if v.default is not None}


class Score(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        raise NotImplementedError("Analysis routine has to be implemented")


class RCScore(Score):
    allowed_str_opts = ["min", "max", "mean", "median", "absmax"]

    #

    def __init__(self, rc_merging="mean"):
        if isinstance(rc_merging, str):
            if rc_merging == "absmax":
                self.rc_merging = self.absmax
            elif rc_merging in self.allowed_str_opts:
                self.rc_merging = lambda x, y: getattr(np, rc_merging)([x, y], axis=0)
        elif callable(rc_merging):
            self.rc_merging = rc_merging
        else:
            raise Exception("rc_merging has to be a callable function of a string: %s" % str(self.allowed_str_opts))

    #

    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        raise NotImplementedError("Analysis routine has to be implemented")

    @staticmethod
    def absmax(x, y, inplace=True):
        if not inplace:
            x = copy.deepcopy(x)
        replace_filt = np.abs(x) < np.abs(y)
        x[replace_filt] = y[replace_filt]
        return x


class Logit(RCScore):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds if preds[k] is not None]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        diffs = np.log(preds["alt"] / (1 - preds["alt"])) - np.log(preds["ref"] / (1 - preds["ref"]))

        if (preds["ref_rc"] is not None) and ((preds["alt_rc"] is not None)):
            diffs_rc = np.log(preds["alt_rc"] / (1 - preds["alt_rc"])) - np.log(preds["ref_rc"] / (1 - preds["ref_rc"]))
            return self.rc_merging(diffs, diffs_rc)
        else:
            return diffs


class LogitAlt(RCScore):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds if preds[k] is not None]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        logits = np.log(preds["alt"] / (1 - preds["alt"]))

        if preds["alt_rc"] is not None:
            logits_rc = np.log(preds["alt_rc"] / (1 - preds["alt_rc"]))
            return self.rc_merging(logits, logits_rc)
        else:
            return logits


class LogitRef(RCScore):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds if preds[k] is not None]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        logits = np.log(preds["ref"] / (1 - preds["ref"]))

        if preds["ref_rc"] is not None:
            logits_rc = np.log(preds["ref_rc"] / (1 - preds["ref_rc"]))
            return self.rc_merging(logits, logits_rc)
        else:
            return logits


class Alt(RCScore):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        alt_out = alt
        if alt_rc is not None:
            alt_out = self.rc_merging(alt, alt_rc)
        return alt_out


class Ref(RCScore):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        ref_out = ref
        if ref_rc is not None:
            ref_out = self.rc_merging(ref, ref_rc)
        return ref_out


class Diff(RCScore):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        diffs = preds["alt"] - preds["ref"]

        if (preds["ref_rc"] is not None) and ((preds["alt_rc"] is not None)):
            diffs_rc = preds["alt_rc"] - preds["ref_rc"]
            return self.rc_merging(diffs, diffs_rc)
        else:
            return diffs


class DeepSEA_effect(RCScore):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds if preds[k] is not None]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        logit_diffs = np.log(preds["alt"] / (1 - preds["alt"])) - np.log(preds["ref"] / (1 - preds["ref"]))
        diffs = preds["alt"] - preds["ref"]

        if (preds["ref_rc"] is not None) and ((preds["alt_rc"] is not None)):
            logit_diffs_rc = np.log(preds["alt_rc"] / (1 - preds["alt_rc"])) - np.log(
                preds["ref_rc"] / (1 - preds["ref_rc"]))
            diffs_rc = preds["alt_rc"] - preds["ref_rc"]
            logit_diffs = self.rc_merging(logit_diffs, logit_diffs_rc)
            diffs = self.rc_merging(diffs, diffs_rc)
            # self.rc_merging(np.abs(logit_diffs) * np.abs(diffs), np.abs(logit_diffs_rc) * np.abs(diffs_rc))

        return np.abs(logit_diffs) * np.abs(diffs)


# if e.g. 'logit' is allowed then also 'logit_alt' is allowed, etc. values here refer to 'scoring_options' keys.
categorical_enable = {
    "__any__": ["diff", "ref", "alt"],
    "logit": ["logit", "logit_ref", "logit_alt", "deepsea_effect"]
}
builtin_default_kwargs = {"rc_merging": "mean"}

scoring_options = {
    # deepsea_effect diff logit_diff logit_ref logit_alt
    # TODO - we should add more options to it: ref, alt, ref_rc, alt_rc
    "ref": Ref,
    "alt": Alt,
    "diff": Diff,
    "logit_ref": LogitRef,
    "logit_alt": LogitAlt,
    "logit": Logit,
    "deepsea_effect": DeepSEA_effect
}

scoring_names = {
    VarEffectFuncType.diff: "diff",
    VarEffectFuncType.ref: "ref",
    VarEffectFuncType.alt: "alt",
    VarEffectFuncType.logit: "logit",
    VarEffectFuncType.logit_ref: "logit_ref",
    VarEffectFuncType.logit_alt: "logit_alt",
    VarEffectFuncType.deepsea_effect: "deepsea_effect",
}


def get_avail_scoring_fns(model):
    if model.postprocessing.variant_effects is None:
        raise Exception("Model deosn't support variant effect prediction according model yaml file.")
    avail_scoring_fns = []  # contains callables
    avail_scoring_fn_def_args = []  # default kwargs for every callable
    avail_scoring_fn_names = []  # contains the labels
    default_scoring_fns = []  # contains the labels of the defaults
    for sf in model.postprocessing.variant_effects.scoring_functions:
        if sf.type is not VarEffectFuncType.custom:
            sn = scoring_names[sf.type]
            sf_obj = scoring_options[sn]
            s_label = sn
            if (sf.name != "") and (sf.name not in scoring_options):
                if sf.name in avail_scoring_fn_names:
                    raise Exception("Scoring function names have to unique in the model yaml file.")
                s_label = sf.name
            def_kwargs = builtin_default_kwargs
        else:
            prefix = ""
            if (sf.name == "") or (sf.name in scoring_options):
                prefix = "custom_"
            s_label = prefix + sf.name
            if s_label in avail_scoring_fn_names:
                raise Exception("Scoring function names have to unique in the model yaml file.")
            if sf.defined_as == "":
                raise Exception("`defined_as` has to be defined for a custom function.")
            file_path, obj_name = tuple(sf.defined_as.split("::"))
            sf_obj = getattr(load_module(file_path), obj_name)
            # check that the scoring function arguments match yaml arguments
            if not getargs(sf_obj) == set(sf.args.keys()):
                raise ValueError("Scoring function arguments: \n{0}\n don't match ".format(set(getargs(sf_obj))) +
                                 "the specification in the dataloader.yaml file:\n{0}".
                                 format(set(sf.args.keys())))

            def_kwargs = None
            if all([(sf.args[k].default is not None) or (sf.args[k].optional) for k in sf.args]):
                def_kwargs = default_kwargs(sf.args)
            if len(sf.args) == 0:
                def_kwargs = {}  # indicates that this callable doesn't accept any arguments for initialisation.

        if s_label in avail_scoring_fn_names:
            raise Exception("Mulitple scoring functions defined with name '%s' in the model yaml file!" % s_label)

        avail_scoring_fn_def_args.append(def_kwargs)
        avail_scoring_fns.append(sf_obj)
        avail_scoring_fn_names.append(s_label)
        if sf.default:
            default_scoring_fns.append(s_label)

    # if no default scoring functions have been set then take all of the above.
    if len(default_scoring_fns) == 0:
        default_scoring_fns = copy.copy(avail_scoring_fn_names)

    # try to complete the set of functions if needed. None of the additional ones will become part of the defaults
    additional_scoring_fn_def_args = []
    additional_scoring_fns = []
    additional_scoring_fn_names = []
    for scoring_fn, def_args in zip(avail_scoring_fns, avail_scoring_fn_def_args):
        scoring_name = None
        for sn in scoring_options:
            if scoring_fn is scoring_options[sn]:
                scoring_name = sn
                break
        if scoring_name is None:
            continue
        categories = [cat for cat in categorical_enable if scoring_name in categorical_enable[cat]]
        for cat in categories:
            for scoring_name in categorical_enable[cat]:
                if (scoring_options[scoring_name] not in avail_scoring_fns) and \
                        (scoring_options[scoring_name] not in additional_scoring_fns):
                    additional_scoring_fn_def_args.append(def_args)
                    additional_scoring_fns.append(scoring_options[scoring_name])
                    s_label = scoring_name
                    if s_label in avail_scoring_fn_names:
                        s_label = "default_" + s_label
                    additional_scoring_fn_names.append(s_label)

    avail_scoring_fns += additional_scoring_fns
    avail_scoring_fn_def_args += additional_scoring_fn_def_args
    avail_scoring_fn_names += additional_scoring_fn_names

    # add the default scoring functions if not already in there
    for scoring_name in categorical_enable["__any__"]:
        if scoring_options[scoring_name] not in avail_scoring_fns:
            avail_scoring_fn_def_args.append(builtin_default_kwargs)
            avail_scoring_fns.append(scoring_options[scoring_name])
            s_label = scoring_name
            if s_label in avail_scoring_fn_names:
                s_label = "default_" + s_label
            avail_scoring_fn_names.append(s_label)
            if len(default_scoring_fns) == 0:
                default_scoring_fns.append(s_label)

    return avail_scoring_fns, avail_scoring_fn_def_args, avail_scoring_fn_names, default_scoring_fns


def get_scoring_fns(model, scoring_fns, scoring_kwargs=None):
    """
    Transform a list of scoring functions and names to a dictionary of scoring functions that are set up with kwargs
    defined in scoring_kwargs.
    
    Arguments
        model: Kipoi Model object
        scoring_fns: A list of scoring functions or strings of scoring functions. 
        scoring_kwargs: Either list of length 0 or a list of dicts (kwargs) with the same length of scoring_fns and the 
        same order. If an entry of scoring_fns is a string then the scoring function will be initialised with the 
        corresponding entry in scoring_kwargs. If an entry of scoring_fns is a callable the corresponding entry in 
        scoring_kwargs will be ignored.
    """
    # get the scoring methods according to the model definition
    avail_scoring_fns, avail_scoring_fn_def_args, avail_scoring_fn_names, \
    default_scoring_fns = get_avail_scoring_fns(model)

    errmsg_scoring_kwargs = "When using `scoring_kwargs` a kwargs dictionary for every entry in " \
                            "`scoring_kwargs` must be given."
    if scoring_kwargs is None:
        scoring_kwargs = []

    dts = {}
    if len(scoring_fns) >= 1:
        # Check if all scoring functions should be used:
        if scoring_fns == ["all"]:
            if len(scoring_kwargs) >= 1:
                raise ValueError("`scoring_kwargs` cannot be defined when `scoring_fns` == ['all']!")
            for arg_iter, k in enumerate(avail_scoring_fn_names):
                si = avail_scoring_fn_names.index(k)
                # get the default kwargs
                kwargs = avail_scoring_fn_def_args[si]
                if kwargs is None:
                    raise ValueError("No default kwargs for scoring function: %s"
                                     " `scoring_fns` == ['all'] cannot be used. "
                                     " Please select the desired scoring functions explicitely and also define "
                                     "`scoring_kwargs`." % (k))
                # instantiate the scoring fn
                dts[k] = avail_scoring_fns[si](**kwargs)
        else:
            # if -k set check that length matches with -s
            if len(scoring_kwargs) >= 1:
                if not len(scoring_fns) == len(scoring_kwargs):
                    raise ValueError(errmsg_scoring_kwargs)
            for arg_iter, k in enumerate(scoring_fns):
                if isinstance(k, six.string_types):
                    # if -s set check is available for model
                    if k in avail_scoring_fn_names:
                        si = avail_scoring_fn_names.index(k)
                        # get the default kwargs
                        kwargs = avail_scoring_fn_def_args[si]
                        # if the user has set scoring function kwargs then load them here.
                        if len(scoring_kwargs) >= 1:
                            # all the {}s in -k replace by their defaults, if the default is None
                            # raise exception with the corrsponding scoring function label etc.
                            if len(scoring_kwargs[arg_iter]) != 0:
                                kwargs = scoring_kwargs[arg_iter]
                        if kwargs is None:
                            raise ValueError("No kwargs were given for scoring function %s"
                                             " with no defaults but required argmuents. "
                                             "Please also define sel_scoring_kwargs." % (k))
                        # instantiate the scoring fn
                        dts[k] = avail_scoring_fns[si](**kwargs)
                    else:
                        logger.warn("Cannot choose scoring function %s. "
                                    "Model only supports: %s." % (k, str(avail_scoring_fn_names)))
                elif callable(k):
                    # TODO: Add tests
                    if not hasattr(k, "__name__"):
                        raise Exception("scoring functions that are passed as callables have to have a __name__ "
                                        "attribute.")
                    dts[k.__name__] = k

    # if -s not set use all defaults
    elif len(default_scoring_fns) != 0:
        for arg_iter, k in enumerate(default_scoring_fns):
            si = avail_scoring_fn_names.index(k)
            kwargs = avail_scoring_fn_def_args[si]
            dts[k] = avail_scoring_fns[si](**kwargs)

    if len(dts) == 0:
        raise Exception("No scoring method was chosen!")

    return dts
