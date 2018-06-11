import kipoi
import pytest
from pytest import raises
from kipoi.components import ModelSchema
from related import from_yaml
import config
import sys

INSTALL_REQ = config.install_req
from kipoi.pipeline import install_model_requirements
import json
from kipoi.utils import cd
from tqdm import tqdm
from kipoi import writers
from kipoi.cli.main import prepare_batch
import os
from kipoi.data_utils import numpy_collate
import numpy as np
import copy

predict_activation_layers = {
    "rbp": "concatenate_6",
    # "pyt": "3"  # two before the last layer
}

MODEL_SCHEMA_EXAMPLES = ["""
inputs:
    seq:
        shape: (4, None)
        doc: One-hot encoded RNA sequence
        special_type: DNASeq
        associated_metadata: ranges
targets:
    binding_site:
        shape: (1, )
        doc: Binding strength
""", """
inputs:
      - name: seq
        shape: (4, 100)
        doc: One-hot encoded RNA sequence
        special_type: DNASeq
        associated_metadata:
          - ranges
targets:
    binding_site:
        shape: (1, )
        doc: Binding strength
""", """
inputs:
    name: seq
    shape: (4, 100)
    doc: One-hot encoded DNA sequence
    special_type: bigwig
targets:
    binding_site:
        shape: (1, )
        doc: Binding strength
        special_type: bigwig
"""]

dummy_dataset = []


def test_get_selector():
    from kipoi.postprocessing.gradient_vis.vis import get_selector
    assert get_selector(1, slice(None, None, -1)) == (slice(None, None, None), slice(None, None, -1), Ellipsis)
    assert get_selector(-1, slice(None, None, -1)) == (Ellipsis, slice(None, None, -1))


def get_example_data(example, layer, writer=None):
    example_dir = "examples/{0}".format(example)
    if INSTALL_REQ:
        install_model_requirements(example_dir, "dir", and_dataloaders=True)

    model = kipoi.get_model(example_dir, source="dir")
    # The preprocessor
    Dataloader = kipoi.get_dataloader_factory(example_dir, source="dir")
    #
    with open(example_dir + "/example_files/test.json", "r") as ifh:
        dataloader_arguments = json.load(ifh)

    for k in dataloader_arguments:
        dataloader_arguments[k] = "example_files/" + dataloader_arguments[k]

    outputs = []
    with cd(model.source_dir):
        dl = Dataloader(**dataloader_arguments)
        it = dl.batch_iter(batch_size=32, num_workers=0)

        # Loop through the data, make predictions, save the output
        for i, batch in enumerate(tqdm(it)):

            # make the prediction
            pred_batch = model.input_grad(batch['inputs'], avg_func="sum", layer=layer,
                                          final_layer=False)
            # write out the predictions, metadata (, inputs, targets)
            # always keep the inputs so that input*grad can be generated!
            output_batch = batch
            output_batch["grads"] = pred_batch
            if writer is not None:
                writer.batch_write(output_batch)
            outputs.append(output_batch)
        if writer is not None:
            writer.close()
    return numpy_collate(outputs)


# @pytest.mark.parametrize("example", list(predict_activation_layers))
def test_gradplotter():
    from kipoi.postprocessing.gradient_vis.vis import GradPlotter, get_selector
    example = "rbp"
    if example in {"rbp", "non_bedinput_model", "iris_model_template"} and sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")

    example_dir = "examples/{0}".format(example)

    output = os.path.realpath(example_dir + "/grad_outputs.hdf5")
    try:
        os.unlink(output)
    except:
        pass

    writer = writers.HDF5BatchWriter(file_path=output)
    get_example_data(example, predict_activation_layers[example], writer=writer)

    gp = GradPlotter.from_hdf5(output, example_dir, source="dir")

    # test get_num_samples
    assert gp.data['inputs']['seq'].shape[0] == gp.get_num_samples("seq")

    # once we have a gp instance:
    exp_ret_fns = [gp._select_ds_dict, gp._select_ds_list, gp._select_ds_ndarray]
    for model_schema_yaml, exp_ret_fn in zip(MODEL_SCHEMA_EXAMPLES, exp_ret_fns):
        schema = ModelSchema.from_config(from_yaml(model_schema_yaml))
        ret_fn, ret_lables = gp._get_ds_extractor(schema.inputs)
        assert ret_fn == exp_ret_fn
        assert ret_lables == ['seq']
    try:
        os.unlink(output)
    except:
        pass


def test__infer_seq_dim():
    from kipoi.postprocessing.gradient_vis.vis import GradPlotter
    assert GradPlotter._infer_seq_dim(3, [1, 3, 2, 4]) == 1
    assert GradPlotter._infer_seq_dim(3, [1, 3, 3, 4]) is None
    assert GradPlotter._infer_seq_dim(3, [1, 1, 1, 4]) is None


def test_verify_model_input():
    import pytest
    from kipoi.postprocessing.gradient_vis.vis import GradPlotter
    gp = GradPlotter({"inputs": [], "grads": []}, "examples/rbp", source="dir")
    with pytest.raises(Exception):
        gp._verify_model_input(None)
    with pytest.warns(None):
        gp = GradPlotter({"inputs": [], "grads": []}, "examples/pyt", source="dir")
        assert gp._verify_model_input(None) == 'input'


class DummyRegionWriter(object):
    def __init__(self):
        self.regions = []
        self.datas = []

    def region_write(self, region, data):
        self.regions.append(region)
        self.datas.append(data)

    def close(self):
        pass


def test_plot():
    from kipoi.postprocessing.gradient_vis.vis import GradPlotter
    # test genomic region subsetting
    example_data = get_example_data("pyt", "3")
    gp = GradPlotter(example_data, "examples/pyt", source="dir")
    # np.random.seed(1)
    # example_data['grads'] = np.random.randn(*example_data['grads'].shape)

    # 1) Test Region subsetting (genomic)
    full_first_region = {k: v[0, 0] for k, v in example_data['metadata']['ranges'].items()}
    subset_region = (full_first_region["start"] + 20, full_first_region["end"] - 20)
    drw = DummyRegionWriter()
    gp.write(0, limit_region_genomic=subset_region, writer_obj=drw)
    assert len(drw.regions) == 1
    assert drw.regions[0]["start"][0] == subset_region[0]
    assert drw.regions[0]["end"][0] == subset_region[1]
    assert drw.datas[0].shape[0] == drw.regions[0]["end"][0] - drw.regions[0]["start"][0]

    # 3) Test whether seq_dims are reduced correctly
    drw = DummyRegionWriter()
    gp.write(0, writer_obj=drw)
    input_grad_0 = gp.data['inputs'][0, ...] * gp.data['grads'][0, ...]
    seq_dim = np.where(np.array(input_grad_0.shape) == full_first_region["end"] - full_first_region["start"])[0][0]
    sel = [slice(None)] * (seq_dim) + [0] + [Ellipsis]
    data_exp_0 = input_grad_0[tuple(sel)].sum()
    assert np.isclose(data_exp_0, drw.datas[0][0])

    # 4) Test additional_plot_fns
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.pyplot.switch_backend('agg')
    example_data_cp = copy.deepcopy(example_data)
    example_data_cp['inputs'] = example_data_cp['inputs'][0, ...]
    example_data_cp['grads'] = example_data_cp['grads'][0, ...]
    gp2 = GradPlotter(example_data_cp, "examples/pyt", source="dir")
    plt.figure()
    ax_obj = plt.subplot(1, 1, 1)

    def dummy_plotfn(chrom, start, end, ax):
        assert chrom == full_first_region["chr"]
        assert start == subset_region[0]
        assert end == subset_region[1]
        assert ax == ax_obj

    gp2.plot(0, limit_region_genomic=subset_region, additional_plot_fns=[dummy_plotfn], ax=ax_obj)
