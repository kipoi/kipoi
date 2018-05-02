from kipoi import readers
from kipoi.postprocessing.variant_effects import ModelInfoExtractor
from kipoi.postprocessing.variant_effects.utils.generic import _get_metadata_name, _get_seq_shape
from kipoi.postprocessing.variant_effects.utils import ReshapeDna
import kipoi
import os
import numpy as np

def grad_x_input(input, grad):
    return input*grad

def get_selector(dim, slice_at_dim):
    """
    Place `slice_at_dim` at dimension `dim` for setting or setting items
    """
    if dim >= 0:
        selector = [slice(None) for i in range(dim)] + [slice_at_dim]  + [Ellipsis]
    else:
        selector = [Ellipsis] + [slice_at_dim] + [slice(None) for i in range((dim)*(-1)-1)]
    selector = tuple(selector)
    return selector

class GradPlotter(object):
    def __init__(self, results_fname, model, source="kipoi"):
        self.data = readers.HDF5Reader.load(results_fname, unflatten=True)

        # TODO: Instead of copying from kipoi.model should we rather have a get_model_descr funcion that is also called from get_model

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
        default_dataloader = kipoi.get_dataloader_factory(default_dataloader_path,
                                                          dl_source)

        self.mie = ModelInfoExtractor(md, default_dataloader)
        self.md =md
        self.dataloader = default_dataloader

        # how can the correct model input be selected
        self.get_dataset, self.model_input_keylist = self._get_ds_extractor(md.schema.inputs)

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


    def plot(self, sample, model_input=None, ax=None, limit_region=None, limit_region_genomic=None, rc_plot = False,
                    transform_fn = grad_x_input, seq_dim = None, additional_plot_fns = None, seq_plotter_obj = None,
                    **heatmap_kwargs):

        def raise_missing_metadata():
            raise Exception("limit_region_genomic can only be used with a `model_input` that has an associated"
                            "metadata field with at least the following entries: 'chrom', 'start', 'end' or type"
                            "GenomicRanges.")
        def get_metadata():
            metadata_field = None
            try:
                metadata_field = _get_metadata_name(self.dataloader, model_input)
            except ValueError:
                raise_missing_metadata()
            mf = self.data['metadata'][metadata_field]
            if not all([el in mf for el in ["chr", "start", "end"]]):
                raise_missing_metadata()
            return mf


        inputs = self.data["inputs"]
        gradients = self.data["preds"]


        if model_input is None:
            if len(self.model_input_keylist) != 1:
                raise Exception("model_input cannot be None for models that have multiple "
                                "inputs: %s"%str(self.model_input_keylist))
            model_input = self.model_input_keylist[0]

        input = self.get_dataset(model_input, inputs)
        gradient = self.get_dataset(model_input, gradients)


        is_onehot_seq = False
        if model_input in self.mie.get_mutatable_inputs(only_one_hot=True):
            is_onehot_seq = True

        mr_start = None
        mr_end = None
        if (limit_region_genomic is not None) and isinstance(limit_region_genomic, tuple):
            mf = get_metadata()
            mr_start = mf["start"][sample]
            mr_end = mf["end"][sample]
            if any([(el < mr_start) or (el > mr_end) for el in list(limit_region_genomic)]):
                raise Exception("`limit_region_genomic` has to lie within: %s" % str([mr_start, mr_end]))
            limit_region = (limit_region_genomic[0] - mr_start, limit_region_genomic[1] - mr_start,)

        import seaborn as sns
        if is_onehot_seq:
            # convert to standard layout
            dna_reshaper = ReshapeDna(_get_seq_shape(self.dataloader, model_input))
            input_reshaped = dna_reshaper.to_standard(input)[sample,...]
            gradient_reshaped = dna_reshaper.to_standard(gradient)[sample,...]

            values = transform_fn(input_reshaped, gradient_reshaped)

            if rc_plot:
                values = values[::-1, ::-1]

            if limit_region is not None:
                values = values[limit_region[0]:limit_region[1], :]

            if seq_plotter_obj is not None:
                mf = get_metadata()
                if mr_start is None:
                    mr_start = mf["start"][sample]
                    mr_end = mf["end"][sample]
                regions = {"chr":[mf["chr"][sample]], "start":[mr_start], "end":[mr_end]}
                seq_plotter_obj.region_write(regions, values)
            else:
                from kipoi.external.concise.seqplotting_deps import seqlogo
                seqlogo(values, ax=ax)
                ax.axes.get_xaxis().set_visible(False)
                sns.despine(trim=True, bottom = True)

        else:
            if rc_plot or (limit_region is not None):
                if seq_dim is None:
                    raise Exception("seq_dim has to be defined if `rc_plot` or `limit_region_genomic` are used.")

            values = transform_fn(input[sample,...], gradient[sample,...])

            if limit_region is not None:
                lr_sel = get_selector(seq_dim, slice(limit_region[0], limit_region[1]))
                values = values.__getitem__(lr_sel)

            if rc_plot:
                rc_sel = get_selector(seq_dim, slice(None, None, -1))
                values = values.__getitem__(rc_sel)

            sns.heatmap(values, ax = ax, **heatmap_kwargs)

        if additional_plot_fns is not None:
            mf = get_metadata()
            for plot_fn in additional_plot_fns:
                start = mf["start"][sample]
                end = mf["end"][sample]
                if limit_region is not None:
                    start = start + limit_region[0]
                    end = start + limit_region[1]
                plot_fn(chrom= mf["chr"][sample], start=start, end=end, ax=ax)





