from kipoi import readers
from kipoi.postprocessing.variant_effects import ModelInfoExtractor
from kipoi.postprocessing.variant_effects.utils.generic import _get_metadata_name, _get_seq_shape
from kipoi.postprocessing.variant_effects.utils import ReshapeDna
import kipoi
import os
import numpy as np
import logging
from kipoi.utils import cd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def grad_x_input(input, grad):
    return input * grad


def get_selector(dim, slice_at_dim):
    """
    Place `slice_at_dim` at dimension `dim` for setting or setting items
    """
    if dim >= 0:
        selector = [slice(None) for i in range(dim)] + [slice_at_dim] + [Ellipsis]
    else:
        selector = [Ellipsis] + [slice_at_dim] + [slice(None) for i in range((dim) * (-1) - 1)]
    selector = tuple(selector)
    return selector


class GradPlotter(object):
    """
    Class for plotting gradients. Results can be loaded from a HDF5 file or directly from returns of
    model.input_grad(...)
    """

    def __init__(self, data, model, source="kipoi", grad_preds=None):
        """
        Arguments:
            data: model input data batch 
            model: model name as used for running `model.input_grad(...)`
            source: model source as used for running `model.input_grad(...)`
            grad_preds: return value of `model.input_grad(...)`. Can alternatively already be present in `data`
            argument under the key `preds`. In that case `grad_preds` may be None.
        """
        self.data = data
        if grad_preds is not None:
            self.data['preds'] = grad_preds
        else:
            assert 'preds' in self.data

        # TODO: Instead of copying from kipoi.model should we rather have a get_model_descr
        # TODO-cont: funcion that is also called from get_model
        # Taken from get_model
        source_name = source
        source = kipoi.config.get_source(source)
        md = source.get_model_descr(model)

        if ":" in md.default_dataloader:
            dl_source, dl_path = md.default_dataloader.split(":")
        else:
            dl_source = source_name
            dl_path = md.default_dataloader

        # allow to use relative and absolute paths for referring to the dataloader
        default_dataloader_path = os.path.join("/" + model, dl_path)[1:]
        # This one loads the model!!
        # default_dataloader = kipoi.get_dataloader_factory(default_dataloader_path,
        #                                                  dl_source)

        # TODO: Is there a nicer way of getting ahold of the dataloader description?
        yaml_path = source.pull_dataloader(default_dataloader_path)
        dataloader_dir = os.path.dirname(yaml_path)
        from kipoi.components import DataLoaderDescription
        with cd(dataloader_dir):
            dl = DataLoaderDescription.load(os.path.basename(yaml_path))
            default_dataloader = dl

        try:
            self.mie = ModelInfoExtractor(md, default_dataloader)
        except:
            logger.warn("Model is not enabled for variant effect prediction hence it is unclear whether there is a DNA "
                        "sequence input, so (automatic) seqlogo plots are not available for this model.")
            self.mie = None
        self.md = md
        self.dataloader = default_dataloader

        # how can the correct model input be selected
        self.get_dataset, self.model_input_keylist = self._get_ds_extractor(md.schema.inputs)

    @classmethod
    def from_hdf5(self, results_fname, model, source="kipoi"):
        """
        Arguments:
            results_fname: HDF5 file produced by running `model.input_grad(...)`. The file has to contain
            also the model input!
            model: model name as used for running `model.input_grad(...)`
            source: model source as used for running `model.input_grad(...)`
        """
        data = readers.HDF5Reader.load(results_fname, unflatten=True)
        return self(data, model=model, source=source)

    def _get_ds_extractor(self, model_input_schema):
        from kipoi.components import ArraySchema
        # return data selection function + labels
        if isinstance(model_input_schema, ArraySchema):
            return self._select_ds_ndarray, [model_input_schema.name]
        elif isinstance(model_input_schema, list):
            return self._select_ds_list, [el.name for el in model_input_schema]
        elif isinstance(model_input_schema, dict):
            return self._select_ds_dict, [el for el in model_input_schema]

    def _select_ds_ndarray(self, index, dataset):
        if index is not None:
            assert (index == 0) or (self.model_input_keylist[0] == index)
        return dataset

    def _select_ds_list(self, index, dataset):
        if isinstance(index, int):
            return dataset[index]
        else:
            int_ind = self.model_input_keylist.index(index)
            return dataset[int_ind]

    def _select_ds_dict(self, index, dataset):
        return dataset[index]

    def _verify_model_input(self, model_input):
        if model_input is None:
            if len(self.model_input_keylist) != 1:
                raise Exception("model_input cannot be None for models that have multiple "
                                "inputs: %s" % str(self.model_input_keylist))
            model_input = self.model_input_keylist[0]
        return model_input

    def get_num_samples(self, model_input=None):
        """
        Get number of samples present in the dataset (size of 0th dimension of the model input).
        
        Arguments:
            model_input: Number of samples will be returned for that model input.
        """
        model_input = self._verify_model_input(model_input)
        input = self.get_dataset(model_input, self.data["inputs"])
        return input.shape[0]

    @staticmethod
    def _infer_seq_dim(len, array_shape):
        sel = np.array(array_shape) == len
        if sel.sum() == 1:
            return np.where(sel)[0][0]
        return None

    def _preprocess_values(self, sample, model_input=None, limit_region=None, limit_region_genomic=None,
                           transform_fn=grad_x_input, seq_dim=None, requires_region_info=False,
                           requires_seq_dim=False):
        def raise_missing_metadata():
            raise Exception("limit_region_genomic can only be used with a `model_input` that has an associated"
                            "metadata field with at least the following entries: 'chrom', 'start', 'end' or type"
                            "GenomicRanges.")

        def get_metadata_cse():
            metadata_field = None
            try:
                metadata_field = _get_metadata_name(self.dataloader, model_input)
            except ValueError:
                raise_missing_metadata()
            mf = self.data['metadata'][metadata_field]
            if not all([el in mf for el in ["chr", "start", "end"]]):
                raise_missing_metadata()
            cse = {k: np.squeeze(mf[k]) for k in ["chr", "start", "end"]}
            for k in ["chr", "start", "end"]:
                cse[k] = np.squeeze(mf[k])
                # if only one sample then squeeze does too much, so correct that:
                if len(cse[k].shape) == 0:
                    cse[k] = np.array([cse[k]])
                if len(cse[k].shape) != 1:
                    raise Exception("Invalid metadata format for field ['%s']['%s'] with shape: %s" %
                                    (metadata_field, k, str(mf[k].shape)))
            return cse

        inputs = self.data["inputs"]
        gradients = self.data["preds"]

        model_input = self._verify_model_input(model_input)

        input = self.get_dataset(model_input, inputs)
        gradient = self.get_dataset(model_input, gradients)

        is_onehot_seq = False
        if (self.mie is not None) and (model_input in self.mie.get_mutatable_inputs(only_one_hot=True)):
            is_onehot_seq = True

        if requires_region_info:
            mf = get_metadata_cse()
            mr_chr = mf["chr"][sample]
            mr_start = mf["start"][sample]
            mr_end = mf["end"][sample]
        else:
            mr_chr = None
            mr_start = None
            mr_end = None

        if (limit_region_genomic is not None):
            if not isinstance(limit_region_genomic, tuple):
                raise Exception("`limit_region_genomic` has to be a tuple of (start, end) genomic coordinates!")
            mf = get_metadata_cse()
            mr_chr = mf["chr"][sample]
            mr_start = mf["start"][sample]
            mr_end = mf["end"][sample]
            if any([(el < mr_start) or (el > mr_end) for el in list(limit_region_genomic)]):
                raise Exception("`limit_region_genomic` has to lie within: %s" % str([mr_start, mr_end]))
            limit_region = (limit_region_genomic[0] - mr_start, limit_region_genomic[1] - mr_start,)
            mr_start, mr_end = limit_region_genomic

        elif (limit_region is not None):
            if not isinstance(limit_region, tuple):
                raise Exception("`limit_region` has to be a tuple of (start, end) array indices!")
            if mr_start is not None:
                mr_end = mr_start + limit_region_genomic[1] + 1
                mr_start = mr_start + limit_region_genomic[0]

        if is_onehot_seq:
            # convert to standard layout
            dna_reshaper = ReshapeDna(_get_seq_shape(self.dataloader, model_input))
            input_reshaped = dna_reshaper.to_standard(input)[sample, ...]
            gradient_reshaped = dna_reshaper.to_standard(gradient)[sample, ...]
            seq_dim = 0

            values = transform_fn(input_reshaped, gradient_reshaped)

            if limit_region is not None:
                values = values[limit_region[0]:limit_region[1], :]

            return values, is_onehot_seq, mr_chr, mr_start, mr_end, seq_dim

        else:
            if requires_seq_dim:
                if seq_dim is None:
                    mf = get_metadata_cse()
                    seq_len = mf["end"][sample] - mf["start"][sample]
                    seq_dim = self._infer_seq_dim(seq_len, input.shape[1:])
                    if seq_dim is None:
                        raise Exception("seq_dim was not defined and could not be inferred for array dimensions %s and"
                                        "sequence length %d. seq_dim is required if `rc_plot` or "
                                        "`limit_region_genomic` or `seq_plotter_obj` on a non-DNAsequence input "
                                        "are used!")

            values = transform_fn(input[sample, ...], gradient[sample, ...])
            if limit_region is not None:
                lr_sel = get_selector(seq_dim, slice(limit_region[0], limit_region[1]))
                values = values.__getitem__(lr_sel)

            return values, is_onehot_seq, mr_chr, mr_start, mr_end, seq_dim

    def plot(self, sample, model_input=None, ax=None, limit_region=None, limit_region_genomic=None, rc_plot=False,
             transform_fn=grad_x_input, seq_dim=None, additional_plot_fns=None, **heatmap_kwargs):
        """
        Plot grad*input for one sample in the data (batch). If the selected model input is tagged as "DNASequence" and 
        the model variant effect prediction activated (here only necessary for parsing model info), then values 
        returned by `transform_fn` are displayed as Seqlogo plots. The default transform_fn is grad*input.
        If the selected model input is not tagged as "DNASequence" or the model is not activated for variant effect
        prediction then a heatmap will be generated with additional `heatmap_kwargs` if given.
        If a heatmap should be produced then the input can only be the batch sample axis + 2D, otherwise the heatmap
        fails.  
        
        Arguments:
            sample: Sample in the batch (integer)
            model_input: Name of the model input that should be plotted (can be omitted for models with only one
            input)
            ax: axis object to be passed to the plotting functions.
            limit_region: Tuple. Limits the plot to a subset in the sequence dimension (seq_dim)
            limit_region_genomic: Tuple. Like `limit_region`, but genomic coordinates - no chromosome
            rc_plot: Reverse-complement the plot. If model_input is not "DNASequence" then only reverse
            transform_fn: Function fn(input, grad). Default is input*grad
            seq_dim: Dimension of the sequence. Used for reversing the order in `rc_plot`. Can be omitted for
            "DNASequence" model inputs. If not given, but needed it will be attempted to be inferred from array
            dimensions and metadata sequence length. 
            additional_plot_fns: List of functions fn(chrom, start, end, ax) that will be executed after the main
            plotting routine 
        """

        requires_region_info = additional_plot_fns is not None

        values, is_onehot_seq, mr_chr, mr_start, mr_end, seq_dim = self._preprocess_values(sample,
                                                                                           model_input=model_input,
                                                                                           limit_region=limit_region,
                                                                                           limit_region_genomic=limit_region_genomic,
                                                                                           transform_fn=transform_fn,
                                                                                           seq_dim=seq_dim,
                                                                                           requires_region_info=requires_region_info,
                                                                                           requires_seq_dim=True)

        if ax is None:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(20, 4))
            ax = plt.subplot(1, 1, 1)

        import seaborn as sns
        if is_onehot_seq:
            # Reverse-complement only for the plot itself
            if rc_plot:
                values = values[::-1, ::-1]

            from kipoi.external.concise.seqplotting_deps import seqlogo
            seqlogo(values, ax=ax)
            ax.axes.get_xaxis().set_visible(False)
            sns.despine(trim=True, bottom=True, ax=ax)

        else:
            if rc_plot:
                rc_sel = get_selector(seq_dim, slice(None, None, -1))
                values = values.__getitem__(rc_sel)

            sns.heatmap(values, ax=ax, **heatmap_kwargs)

        if (additional_plot_fns is not None):
            for plot_fn in additional_plot_fns:
                plot_fn(chrom=mr_chr, start=mr_start, end=mr_end, ax=ax)

    def write(self, sample, writer_obj, model_input=None, limit_region=None, limit_region_genomic=None,
              transform_fn=grad_x_input, seq_dim=None):
        """
        Write grad*input for one sample in the data (batch). If the selected model input is tagged as "DNASequence" and 
        the model variant effect prediction activated (here only necessary for parsing model info), then values 
        returned by `transform_fn` are stored as data in the `writer_obj`. The default transform_fn is grad*input.
        If the selected model input is not tagged as "DNASequence" or the model is not activated for variant effect
        prediction then higher-dimensional data will be compressed into a 1D representation by summing over all axes
        except for the one defined in `seq_dim`. 

        Arguments:
            sample: Sample in the batch (integer)
            writer_obj: Is a instance of a subclass of "RegionWriter". "region_write()" will be called once per
            call of this function.
            model_input: Name of the model input that should be written (can be omitted for models with only one
            input)
            limit_region: Tuple. Limits the values to a subset in the sequence dimension (seq_dim)
            limit_region_genomic: Tuple. Like `limit_region`, but genomic coordinates - no chromosome
            transform_fn: Function fn(input, grad). Default is input*grad
            seq_dim: Dimension of the sequence. Used for reversing the order in `rc_plot`. Can be omitted for
            "DNASequence" model inputs. If not given, but needed it will be attempted to be inferred from array
            dimensions and metadata sequence length. 
        """
        values, is_onehot_seq, mr_chr, mr_start, mr_end, seq_dim = self._preprocess_values(sample,
                                                                                           model_input=model_input,
                                                                                           limit_region=limit_region,
                                                                                           limit_region_genomic=limit_region_genomic,
                                                                                           transform_fn=transform_fn,
                                                                                           seq_dim=seq_dim,
                                                                                           requires_region_info=True,
                                                                                           requires_seq_dim=True)

        regions = {"chr": [mr_chr], "start": [mr_start], "end": [mr_end]}

        if is_onehot_seq:
            writer_obj.region_write(regions, values)


        else:
            # now compress them down by summation so that only the seq_dim is left
            values_summed = values
            for dim in range(0, len(values.shape)):
                if dim != seq_dim:
                    values_summed = np.sum(values_summed, axis=dim, keepdims=True)
            values_summed = np.squeeze(values_summed)
            writer_obj.region_write(regions, values_summed)
