
import kipoi
from kipoi.components import ModelDescription, DataLoaderDescription
from related import from_yaml
from kipoi.postprocessing.utils.generic import ModelInfoExtractor, OneHotSequenceMutator, DNAStringSequenceMutator, ReshapeDnaString, ReshapeDna

dataloader_yaml = """
type: Dataset
defined_as: dataloader.py::SeqDistDataset
args:
    intervals_file:
        doc: tsv file with `chrom start end id score strand`
        type: str
        example: example_files/intervals.tsv
info:
    authors:
        - name: Roman Kreuzhuber
          github: krrome
    doc: DUMMY
output_schema:
    inputs:
        seq_a:
            shape: (101, 4)
            special_type: DNASeq
            doc: One-hot encoded RNA sequence
            associated_metadata: ranges
        seq_b:
            shape: (%s)
            special_type: DNAStringSeq
            doc: DNA sequence as a string
            associated_metadata: ranges
        seq_c:
            shape: (101, 4)
            special_type: DNASeq
            doc: One-hot encoded RNA sequence
            associated_metadata: ranges_b
        something:
            shape: (1, 10)
            doc: Something
    targets:
        shape: (1, )
        doc: Measured binding strength
    metadata:
        ranges:
            type: GenomicRanges
            doc: Ranges describing inputs.seq_a and inputs.seq_b
        ranges_b:
            type: GenomicRanges
            doc: Ranges describing inputs.seq_c
postprocessing:
    variant_effects:
        bed_input:
            - intervals_file
"""

model_yaml = """
type: keras
args:
    arch: model_files/model.json
    weights: model_files/weights.h5
    custom_objects: model/custom_keras_objects.py
default_dataloader: . # path to the directory
info:
    authors:
        - name: Roman Kreuzhuber
          github: krrome
    doc: DUMMY
schema:
    inputs:
        seq_a:
            shape: (101, 4)
            special_type: DNASeq
            doc: One-hot encoded RNA sequence
        seq_b:
            shape: (%s)
            special_type: DNAStringSeq
            doc: DNA sequence as a string
        seq_c:
            shape: (101, 4)
            special_type: DNASeq
            doc: One-hot encoded RNA sequence
        something:
            shape: (1, 10)
            doc: Something
    targets:
        shape: (1, )
        doc: Predicted
postprocessing:
    variant_effects:
          seq_input:
            - seq_a
            - seq_b
            - seq_c
          %s
"""


supports_simple_rc_str = """use_rc: True
"""


def test_ModelDescription():
    for rc_support in [True, False]:
        seq_string_shape = ""
        if rc_support:
            ssrs = supports_simple_rc_str
        else:
            ssrs = ""
        model = ModelDescription.from_config(from_yaml(model_yaml % (seq_string_shape, ssrs)))
        dataloader = DataLoaderDescription.from_config(from_yaml(dataloader_yaml % (seq_string_shape)))
        mi = ModelInfoExtractor(model, dataloader)
        assert mi.use_seq_only_rc == rc_support
        assert all([isinstance(mi.seq_input_mutator[sl], OneHotSequenceMutator) for sl in ["seq_a", "seq_c"]])
        assert all([isinstance(mi.seq_input_mutator[sl], DNAStringSequenceMutator) for sl in ["seq_b"]])
        assert all([mi.seq_input_metadata[sl] == "ranges" for sl in ["seq_a", "seq_b"]])
        assert all([mi.seq_input_metadata[sl] == "ranges_b" for sl in ["seq_c"]])
        assert all([isinstance(mi.seq_input_array_trafo[sl], ReshapeDna) for sl in ["seq_a", "seq_c"]])
        assert all([isinstance(mi.seq_input_array_trafo[sl], ReshapeDnaString) for sl in ["seq_b"]])

