import abc
from abc import abstractmethod
import logging
import copy
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SequenceMutator(object):
    __metaclass__ = abc.ABCMeta

    @abstractmethod
    def __call__(self, input_set, preproc_conv, allele, s_dir):
        """
        Process sequence object `input_set` according to information given in the `preproc_conv` dataframe of which
        the column with name set in argument `allele` is used. `s_dir` defines the output sequence direction: 'fwd'
        or 'rc'. The DNA sequence will then be mutated accordingly.
        """
        raise NotImplementedError("This functionality has to be implemented in the specific subclasses")


def _modify_bases(seq_obj, lines, pos, base, is_rc, return_ref_warning=False):
    # Assumes a fixed order of ACGT and requires one-hot encoding
    assert lines.shape[0] == pos.shape[0]
    assert base.shape[0] == pos.shape[0]
    alphabet = np.array(['A', "C", "G", "T"])
    base_sel = np.where(alphabet[None, :] == base[:, None])
    base_sel_idx = (np.zeros(base.shape[0], dtype=np.int) - 1)
    base_sel_idx[base_sel[0]] = base_sel[1]
    pos = copy.deepcopy(pos)
    # get bases that are not in the alphabet
    known_bases = base_sel_idx != -1
    if is_rc.sum() != 0:
        # is_rc is given relative to the input data, so it as to be transferred back into the variant definition order.
        pos[is_rc] = seq_obj.shape[1] - pos[is_rc] - 1
        base_sel_idx[is_rc] = alphabet.shape[0] - base_sel_idx[is_rc] - 1
    warn_lines = []
    if return_ref_warning:
        wrong_ref = seq_obj[lines[known_bases], pos[known_bases], base_sel_idx[known_bases]] != 1
        warn_lines = np.array(lines[known_bases])[wrong_ref].tolist()
    # Reset the base which was there from the preprocessor
    # Reset the unknown bases just to 0
    seq_obj[lines, pos, :] = 0
    # Set the allele
    seq_obj[lines[known_bases], pos[known_bases], base_sel_idx[known_bases]] = 1
    return warn_lines


def rc_str(dna):
    """Reverse complement a string
    """
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    ks = list(complement.keys())
    for k in ks:
        complement[k.lower()] = complement[k].lower()
    return ''.join([complement[base] for base in dna[::-1]])


def _modify_single_string_base(seq_obj, pos, base, is_rc):
    """modify the single base of a string"""
    if is_rc:
        pos = len(seq_obj) - pos - 1
        base = rc_str(base)
    ret_obj = seq_obj[:pos] + base + seq_obj[(pos + 1):]
    return ret_obj


class OneHotSequenceMutator_OLD(SequenceMutator):
    def __init__(self, array_trafo=None):
        # from the model info object guess
        self.array_trafo = array_trafo

    def __call__(self, input_set, preproc_conv, allele, s_dir):
        """
        Process sequence object `input_set` according to information given in the `preproc_conv` dataframe of which
        the column with name set in argument `allele` is used. `s_dir` defines the output sequence direction: 'fwd'
        or 'rc'. The DNA sequence will then be mutated accordingly. `array_trafo` objects are used to bring non-
        (N,seq_len, 4) arrays into the that shape and convert it back.
        """
        # make sure the sequence objects have the correct length (acording to the ranges specifications)
        if self.array_trafo is not None:
            input_set = self.array_trafo.to_standard(input_set)
        # only keep lines that
        preproc_conv_mutate = preproc_conv.query("do_mutate")
        exp_seq_lens = (preproc_conv_mutate["end"] - preproc_conv_mutate["start"] + 1).unique()
        assert exp_seq_lens.shape[0] == 1
        assert input_set.shape[1] == exp_seq_lens[0]
        assert preproc_conv_mutate["strand"].isin(["+", "-", "*", "."]).all()
        # Modify bases according to allele
        get_warnings = allele == "ref"
        ref_warnings = _modify_bases(seq_obj=input_set,
                                     lines=preproc_conv_mutate["pp_line"].values,
                                     pos=preproc_conv_mutate["varpos_rel"].values.astype(np.int),
                                     base=preproc_conv_mutate[allele].str.upper().values,
                                     is_rc=preproc_conv_mutate["strand"].values == "-",
                                     return_ref_warning=get_warnings)
        for ppl in ref_warnings:
            pcl = np.where(preproc_conv["pp_line"].values == ppl)[0][0]
            vstr = "".join([['A', 'C', 'G', 'T', 'N'][x.argmax() if (x.sum() != 0) and
                                                                    np.all(np.in1d(x, np.arange(0, 4))) else 4] for x in
                            input_set[pcl, ...]])
            logger.warn("Variant reference allele is not the allele present in sequence for:\n%s\n"
                        "Sequence:\n%s" % (str(preproc_conv.iloc[pcl]), vstr))
        # generate reverse complement if needed
        if s_dir == "rc":
            input_set = input_set[:, ::-1, ::-1]
        if self.array_trafo is not None:
            input_set = self.array_trafo.from_standard(input_set)
        return input_set


class OneHotSequenceMutator(SequenceMutator):
    def __init__(self, array_trafo=None):
        # from the model info object guess
        self.array_trafo = array_trafo

    def __call__(self, input_set, variant_loc, allele, s_dir):
        """
        Process sequence object `input_set` according to information given in the `preproc_conv` dataframe of which
        the column with name set in argument `allele` is used. `s_dir` defines the output sequence direction: 'fwd'
        or 'rc'. The DNA sequence will then be mutated accordingly. `array_trafo` objects are used to bring non-
        (N,seq_len, 4) arrays into the that shape and convert it back.
        """
        # make sure the sequence objects have the correct length (acording to the ranges specifications)
        if self.array_trafo is not None:
            input_set = self.array_trafo.to_standard(input_set)
        # only keep lines that
        variant_loc_mutate = variant_loc.subset_to_mutate()
        exp_seq_lens = np.unique(variant_loc_mutate.get_seq_lens())
        assert exp_seq_lens.shape[0] == 1
        assert input_set.shape[1] == exp_seq_lens[0]
        assert variant_loc_mutate.strand_vals_valid()
        # Modify bases according to allele
        get_warnings = allele == "ref"
        ref_warnings = _modify_bases(seq_obj=input_set,
                                     lines=variant_loc_mutate.get("pp_line"),
                                     pos=variant_loc_mutate.get("varpos_rel").astype(np.int),
                                     base=variant_loc_mutate.get(allele, trafo=lambda x: x.upper()),
                                     is_rc=variant_loc_mutate.get("strand") == "-",
                                     return_ref_warning=get_warnings)
        for ppl in ref_warnings:
            pcl = np.where(variant_loc.get("pp_line") == ppl)[0][0]
            vstr = "".join([['A', 'C', 'G', 'T', 'N'][x.argmax() if (x.sum() != 0) and
                                                                    np.all(np.in1d(x, np.arange(0, 4))) else 4] for x in
                            input_set[pcl, ...]])
            logger.warn("Variant reference allele is not the allele present in sequence for:\n%s\n"
                        "Sequence:\n%s" % (str(variant_loc.to_df().iloc[pcl]), vstr))
        # generate reverse complement if needed
        if s_dir == "rc":
            input_set = input_set[:, ::-1, ::-1]
        if self.array_trafo is not None:
            input_set = self.array_trafo.from_standard(input_set)
        return input_set


class DNAStringSequenceMutator_OLD(SequenceMutator):
    def __init__(self, array_trafo=None):
        self.array_trafo = array_trafo

    def __call__(self, input_set, preproc_conv, allele, s_dir):
        """
        Process sequence object `input_set` according to information given in the `preproc_conv` dataframe of which
        the column with name set in argument `allele` is used. `s_dir` defines the output sequence direction: 'fwd'
        or 'rc'. The DNA sequence will then be mutated accordingly.

        The input sequence object must be according to the definition in the dataloader.
        """
        # input_set has to be <list(<str>)> which is achieved by the `array_trafo`.
        if self.array_trafo is not None:
            input_set = self.array_trafo.to_standard(input_set)
        # Every input sequence may have different length, check that ranges defition matches the input.
        seq_lens = np.array([len(el) for el in input_set])
        exp_seq_lens = np.array(preproc_conv["end"] - preproc_conv["start"] + 1)
        assert np.all(seq_lens == exp_seq_lens)
        strands = preproc_conv.query("do_mutate")["strand"]
        assert strands.isin(["+", "-", "*", "."]).all()
        # Modify bases according to allele.
        output_set = []
        for pcl, l in enumerate(preproc_conv["pp_line"].values):
            if preproc_conv["do_mutate"].values[pcl]:
                output_set.append(_modify_single_string_base(input_set[l],
                                                             pos=int(preproc_conv["varpos_rel"].values[pcl]),
                                                             base=preproc_conv[allele].values[pcl],
                                                             is_rc=preproc_conv["strand"].values[pcl] == "-"))
                if allele == "ref":
                    is_rc = preproc_conv["strand"].values[pcl] == "-"
                    base = preproc_conv[allele].values[pcl]
                    vstr = input_set[l]
                    if is_rc:
                        vstr = rc_str(input_set[l])
                    if vstr[int(preproc_conv["varpos_rel"].values[pcl])].upper() != base.upper():
                        logger.warn("Variant reference allele is not the allele present in sequence for:\n%s\n"
                                    "Sequence:\n%s" % (str(preproc_conv.iloc[pcl]), str(input_set[l])))
            else:
                output_set.append(input_set[l])
        # subset to the lines that have been identified
        if len(output_set) != preproc_conv.shape[0]:
            raise Exception("Mismatch between requested and generated DNA sequences.")
        # generate reverse complement if needed
        if s_dir == "rc":
            output_set = [rc_str(el) for el in output_set]
        if self.array_trafo is not None:
            output_set = self.array_trafo.from_standard(output_set)
        return output_set


class DNAStringSequenceMutator(SequenceMutator):
    def __init__(self, array_trafo=None):
        self.array_trafo = array_trafo

    def __call__(self, input_set, variant_loc, allele, s_dir):
        """
        Process sequence object `input_set` according to information given in the `preproc_conv` dataframe of which
        the column with name set in argument `allele` is used. `s_dir` defines the output sequence direction: 'fwd'
        or 'rc'. The DNA sequence will then be mutated accordingly.

        The input sequence object must be according to the definition in the dataloader.
        """
        # input_set has to be <list(<str>)> which is achieved by the `array_trafo`.
        if self.array_trafo is not None:
            input_set = self.array_trafo.to_standard(input_set)
        # Every input sequence may have different length, check that ranges defition matches the input.
        variant_loc_mutate = variant_loc.subset_to_mutate()
        seq_lens = np.array([len(el) for el in input_set])
        exp_seq_lens = variant_loc.get_seq_lens()
        assert np.all(seq_lens == exp_seq_lens)
        assert variant_loc_mutate.strand_vals_valid()
        # Modify bases according to allele.
        output_set = []
        for pcl, l in enumerate(variant_loc["pp_line"]):
            if variant_loc["do_mutate"][pcl]:
                output_set.append(_modify_single_string_base(input_set[l],
                                                             pos=int(variant_loc["varpos_rel"][pcl]),
                                                             base=variant_loc[allele][pcl],
                                                             is_rc=variant_loc["strand"][pcl] == "-"))
                if allele == "ref":
                    is_rc = variant_loc["strand"][pcl] == "-"
                    base = variant_loc[allele][pcl]
                    vstr = input_set[l]
                    if is_rc:
                        vstr = rc_str(input_set[l])
                    if vstr[int(variant_loc["varpos_rel"][pcl])].upper() != base.upper():
                        logger.warn("Variant reference allele is not the allele present in sequence for:\n%s\n"
                                    "Sequence:\n%s" % (str(variant_loc.to_df().iloc[pcl]), str(input_set[l])))
            else:
                output_set.append(input_set[l])
        # subset to the lines that have been identified
        if len(output_set) != variant_loc.num_entries():
            raise Exception("Mismatch between requested and generated DNA sequences.")
        # generate reverse complement if needed
        if s_dir == "rc":
            output_set = [rc_str(el) for el in output_set]
        if self.array_trafo is not None:
            output_set = self.array_trafo.from_standard(output_set)
        return output_set
