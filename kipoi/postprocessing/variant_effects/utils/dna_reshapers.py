import numpy as np
from collections import OrderedDict
import copy
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ReshapeDnaString(object):
    def __init__(self, input_shape):
        if len(input_shape) == 0:
            self.format_style = "string"
            self.seq_len = None
        elif len(input_shape) == 1 and (input_shape[0] == 1):
            self.format_style = "string_in_vect"
            self.seq_len = None
        elif len(input_shape) == 1 and (input_shape[0] > 1):
            self.format_style = "string_as_vect"
            # Can either be a numerical value or None
            self.seq_len = input_shape[0]
        else:
            raise Exception("String output definition not recognized in array string converter!")
        self.single_sample_no_batch_axis = False
        self.input_shape = input_shape

    def get_seq_len(self):
        return self.seq_len

    def to_standard(self, arr):
        if len(arr.shape) == len(self.input_shape):
            arr = arr[None, ...]
            logger.warn("Prepending missing batch axis to input shape %s." % str(self.input_shape))
            self.single_sample_no_batch_axis = True
        elif arr.shape[0] == 1:
            self.single_sample_no_batch_axis = False
        if self.format_style == "string":
            return [str(el) for el in arr]
        elif self.format_style == "string_in_vect":
            return [str(el[0]) for el in arr]
        elif self.format_style == "string_as_vect":
            return ["".join(el.tolist()) for el in arr]

    def from_standard(self, arr):
        if self.format_style == "string":
            arr = np.array(arr)
        elif self.format_style == "string_in_vect":
            arr = np.array(arr)[:, None]
        elif self.format_style == "string_as_vect":
            arr = np.array([list(el) for el in arr])
        if (arr.shape[0] == 1) and self.single_sample_no_batch_axis:
            arr = arr[0, ...]
        return arr


class OutputReshaper(object):
    def __init__(self, model_target_schema, group_delim="."):
        self.model_target_schema = model_target_schema
        self.standard_dict_order = None  # This one is used to always produce the same order of outputs for a dict
        # extract output labels correctly.
        if isinstance(model_target_schema, dict):
            anno = {}
            # Reproducible dict output order:
            self.standard_dict_order = list(model_target_schema.keys())
            if not isinstance(model_target_schema, OrderedDict):
                self.standard_dict_order = sorted(self.standard_dict_order)
            for k in model_target_schema:
                group_name = str(k)
                out_group = model_target_schema[k]
                if out_group.name is not None:
                    group_name = out_group.name
                group_labels = [group_name + group_delim + label for label in self.get_column_names(out_group)]
                anno[k] = np.array(group_labels)
        elif isinstance(model_target_schema, list):
            anno = []
            for i, out_group in enumerate(model_target_schema):
                group_name = str(i)
                if out_group.name is not None:
                    group_name = out_group.name
                group_labels = [group_name + group_delim + label for label in self.get_column_names(out_group)]
                anno.append(np.array(group_labels))
        else:
            anno = self.get_column_names(model_target_schema)
        self.anno = anno

    def get_flat_labels(self):
        if isinstance(self.anno, dict):
            labels = []
            for k in self.standard_dict_order:
                labels.append(self.anno[k])
            flat_labels = np.concatenate(labels, axis=0)
        elif isinstance(self.anno, list):
            flat_labels = np.concatenate(self.anno, axis=0)
        else:
            flat_labels = self.anno
        return flat_labels

    @staticmethod
    def ensure_2dim(arr):
        # These are fixes to when the model output definition is not strictly fulfilled by the model
        if len(arr.shape) == 1:
            arr = arr[:, None]
        elif (len(arr.shape) == 3) and (arr.shape[2] == 1):
            arr = arr[..., 0]
        return arr

    def flatten(self, ds):
        if isinstance(ds, dict):
            if not isinstance(self.anno, dict):
                raise Exception("Error in model output defintion: Model definition is"
                                "of type %s but predictions are of type %s!" % (str(type(ds)), str(type(self.anno))))
            outputs = []
            labels = []
            for k in self.standard_dict_order:
                arr = self.ensure_2dim(ds[k])
                assert (arr.shape[1] == self.anno[k].shape[0])
                outputs.append(arr)
                labels.append(self.anno[k])
            flat = np.concatenate(outputs, axis=1)
            flat_labels = np.concatenate(labels, axis=0)
        elif isinstance(ds, list):
            if not isinstance(self.anno, list):
                raise Exception("Error in model output defintion: Model definition is"
                                "of type %s but predictions are of type %s!" % (str(type(ds)), str(type(self.anno))))
            assert len(ds) == len(self.anno)
            ds = [self.ensure_2dim(el) for el in ds]
            flat = np.concatenate(ds, axis=1)
            flat_labels = np.concatenate(self.anno, axis=0)
        else:
            flat = self.ensure_2dim(ds)
            flat_labels = self.anno
        assert flat.shape[1] == flat_labels.shape[0]
        return flat, flat_labels

    @staticmethod
    def get_column_names(arrayschema_obj):
        if arrayschema_obj.column_labels is not None:
            ret = np.array(arrayschema_obj.column_labels)
        else:
            res_shape = [dim for dim in arrayschema_obj.shape if dim is not None]
            if len(res_shape) > 1:
                raise NotImplementedError(
                    "Don't know how to deal with multi-dimensional model target %s" % str(arrayschema_obj))
            # if res_shape[0] == 1:
            #    ret = np.array([""])
            # else:
            ret = np.arange(res_shape[0]).astype(np.str)
        return ret


class ReshapeDna(object):
    def __init__(self, in_shape):
        in_shape = np.array(in_shape)
        # None can only occur once and then it can only be the seqlen
        none_pos = np.in1d(in_shape, None)
        if np.sum(none_pos) > 1:
            raise Exception("At maximum one occurence of 'None' is allowed in the model input shape!"
                            "This dimension is then automatically assumed to be the 'seq_len' dimension.")
        # if np.any(none_pos) and (np.where(none_pos)[0][0] != 0):
        #    raise Exception("Unexpected 'None' shape in other dimension than the first!")
        # else:
        #    in_shape = in_shape[~none_pos]
        #
        self.in_shape = in_shape
        dummy_dimensions = in_shape == 1
        #
        # Imitate np.squeeze() in self.to_standard()
        in_shape = in_shape[in_shape != 1]
        #
        if in_shape.shape[0] != 2:
            raise Exception("Invalid DNA shape definition definition!")
        #
        one_hot_dim = in_shape == 4
        if one_hot_dim.sum() != 1:
            raise Exception("Could not find 1-hot encoded dimension!")
        #
        seq_len_dim = (in_shape != 1) & (~one_hot_dim)
        if seq_len_dim.sum() != 1:
            raise Exception("Could not find sequence length dimension!")
        #
        dummy_dimensions = np.where(dummy_dimensions)[0]
        one_hot_dim = np.where(one_hot_dim)[0][0]
        seq_len_dim = np.where(seq_len_dim)[0][0]
        #
        self.dummy_dimensions = dummy_dimensions
        self.one_hot_dim = one_hot_dim
        self.seq_len_dim = seq_len_dim
        self.seq_len = in_shape[seq_len_dim]
        self.reshape_needed = True
        if (len(dummy_dimensions) == 0) and (one_hot_dim == 1) and (seq_len_dim == 0):
            self.reshape_needed = False

        # Set default that even for a single sample a batch axis is prepended
        self.single_sample_no_batch_axis = False

    def get_seq_len(self):
        return self.seq_len

    def to_standard(self, in_array):
        """
        :param in_array: has to have the sequence samples in the 0th dimension
        :return:
        """
        #
        # if there is no "batch" axis prepended to the array then add it, but remember
        # that as a rule for `from_standard`
        if (len(in_array.shape) - len(self.in_shape)) == 0:
            in_array = in_array[None, ...]
            logger.warn("Prepending missing batch axis to input shape %s." % str(self.in_shape))
            self.single_sample_no_batch_axis = True

        # if there is a "batch" axis prepended to the array then remember that as a rule
        elif in_array.shape[0] == 1:
            self.single_sample_no_batch_axis = False

        # is there an actual sequence sample axis?
        additional_axis = len(in_array.shape) - len(self.in_shape)
        in_shape = self.in_shape

        # replace the None with the sequence length here:
        if (additional_axis == 1):
            if self.seq_len is None:
                in_shape = copy.deepcopy(in_shape)
                none_dim = [i for i, el in enumerate(self.in_shape) if el is None][0]
                in_shape[none_dim] = in_array.shape[none_dim + 1]

        # Raise an exception if the input array does not agree with the specifications in the model input schema
        if (additional_axis != 1) or (in_array.shape[1:] != tuple(in_shape)):
            raise Exception("General array mismatch! Given: %s Expecting: %s" % (str(in_array.shape),
                                                                                 "([N], %s)" + ", ".join(
                                                                                     in_shape.astype(str).tolist())))
        #
        if not self.reshape_needed:
            return in_array
        squeezed = in_array
        #
        # Iterative removal of dummy dimensions has to start from highest dimension
        for d in sorted(self.dummy_dimensions)[::-1]:
            squeezed = np.squeeze(squeezed, axis=d + additional_axis)
        # check that the shape is now as expected:
        one_hot_dim_here = additional_axis + self.one_hot_dim
        seq_len_dim_here = additional_axis + self.seq_len_dim
        if squeezed.shape[one_hot_dim_here] != 4:
            raise Exception("Input array does not follow the input definition!")
        #
        if (self.seq_len is not None) and (squeezed.shape[seq_len_dim_here] != self.seq_len):
            raise Exception("Input array sequence length does not follow the input definition!")
        #
        if self.one_hot_dim != 1:
            assert (self.seq_len_dim == 1)  # Anything else would be weird...
            squeezed = squeezed.swapaxes(one_hot_dim_here, seq_len_dim_here)
        return squeezed

    def from_standard(self, in_array):
        if not self.reshape_needed:
            if (in_array.shape[0] == 1) and self.single_sample_no_batch_axis:
                in_array = in_array[0, ...]
            return in_array
        #
        assumed_additional_axis = 1
        #
        seq_len = self.seq_len
        # Infer sequence length from the standardised array if not specified in the first place
        if seq_len is None:
            seq_len = in_array.shape[1]

        if in_array.shape[1:] != (seq_len, 4):
            raise Exception("Input array doesn't agree with standard format (n_samples, seq_len, 4)!")
        #
        one_hot_dim_here = assumed_additional_axis + self.one_hot_dim
        seq_len_dim_here = assumed_additional_axis + self.seq_len_dim
        if self.one_hot_dim != 1:
            in_array = in_array.swapaxes(one_hot_dim_here, seq_len_dim_here)
        #
        if len(self.dummy_dimensions) != 0:
            for d in self.dummy_dimensions:
                in_array = np.expand_dims(in_array, d + assumed_additional_axis)
        # If single sample and the convention seems to require no prepended batch axis then remove it.
        if (in_array.shape[0] == 1) and self.single_sample_no_batch_axis:
            in_array = in_array[0, ...]
        return in_array
