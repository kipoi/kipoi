import os
import numpy as np

from kipoi.kipoimodeldescription import KipoiModelSchema, Dependencies, KipoiModelTest, KipoiModelInfo, KipoiModelDescription, Author

def get_labels():
    labels = []
    path_to_labels = os.path.join(os.path.dirname(__file__), "target_labels.txt")
    with open(path_to_labels, "r") as fh:
        labels = [l.rstrip() for l in fh]


args = {
        'module_file': 'pretrained_model_reloaded_th.py',
        'module_obj': 'model',
        'weights': {
            'md5': '4878981d84499eb575abd0f3b45570d3',
            'url': 'https://zenodo.org/record/1466068/files/pretrained_model_reloaded_th.pth?download=1'
        }
    }
default_dataloader = {
  'defined_as': 'kipoiseq.dataloaders.SeqIntervalDl',
  'default_args': {
    'alphabet_axis': 0,
    'auto_resize_len': 600,
    'dtype': np.float32,
    'dummy_axis': 2,
  }
}
schema = KipoiModelSchema(
    inputs = 
    {
        'name': 'seq',
        'doc': 'DNA sequence',
        'shape': (4,600,1),
        'special_type': 'DNASeq',
        'associated_metadata': 'ranges',
    },
    targets = 
    {   
        'column_labels': get_labels(),
        'doc': 'Probability of accessible chromatin in 164 cell types',
        'name': 'DHS_probs',
        'shape': (164, )
    }

)
dependencies = Dependencies(conda=('python=3.6', 'h5py=2.10.0', '_pytorch_select=0.2=gpu_0', 'pytorch=1.3.1=cuda100py36h53c1284_0',
                                   'pip=20.3.3', 'pysam=0.15.3', 'cython=0.29.23'),
                            pip=('kipoiseq'))
test = KipoiModelTest(
    expect={
        'url': 'https://s3.eu-central-1.amazonaws.com/kipoi-models/predictions/14f9bf4b49e21c7b31e8f6d6b9fc69ed88e25f43/Basset/predictions.h5',
        'md5': '9df59f9899b27e65ab95426cb9557ad3'
        }
)
doc = "This is the Basset model published by David Kelley converted to pytorch by \
    \ Roman Kreuzhuber. It categorically predicts probabilities of accesible genomic \
    \ regions in  164 cell types (ENCODE project and Roadmap Epigenomics Consortium). Data was generated using DNAse-seq. The sequence \
    \ length the model uses as input is 600bp. The input of the tensor has to be (N, \
    \ 4, 600, 1) for N samples, 600bp window size and 4 nucleotides. Per sample, 164 \
    \ probabilities of accessible chromatin will be predicted. \n"
trained_on =  "From 2,071,886 total sites, 71,886 randomly reserved for testing and 70,000 for validation, leaving 1,930,000 for training."

info = KipoiModelInfo(authors=(Author("David R. Kelley", "davek44"), ), doc=doc, trained_on=trained_on, 
                        cite_as="https://doi.org/10.1101/gr.200535.115", contributors=(Author("Roman Kreuzhuber", "krrome"), ), 
                        name='Basset', tags=('DNA accessibility'), version='0.1.0')

description = KipoiModelDescription(args=args, schema=schema, info=info, type='pytorch', default_dataloader=default_dataloader,
                                    dependencies=dependencies, test=test)