from kipoi.kipoimodeldescription import KipoiModelSchema, Dependencies, KipoiModelTest, KipoiModelInfo, KipoiModelDescription, Author
from kipoi.model import BaseModel
from tensorflow.keras.models import load_model
import numpy as np


class APARENTModel(BaseModel):

    def __init__(self, weights):
        self.weights = weights
        self.model = load_model(weights)

    def _predict(self, inputs):
        batch_size = inputs.shape[0]

        input_1 = np.expand_dims(inputs, -1)
        input_2 = np.zeros([batch_size, 13])
        input_3 = np.ones([batch_size, 1])

        _, pred = self.model.predict_on_batch([input_1, input_2, input_3])

        site_props = pred[:, :-1]
        distal_prop = pred[:, -1]
        return site_props, distal_prop

    def predict_on_batch(self, inputs):
        site_props, distal_prop = self._predict(inputs)

        return {
            "distal_prop": distal_prop,
            "site_props": site_props,
        }


args = {
        'weights': {
            'md5': '31902fb40125679e655b8b6d2747ada7',
            'url': 'https://github.com/johli/aparent/raw/8a884f0bc4073ed0edd588f71b61a5be4a37e831/saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5'
        }
    }
schema = KipoiModelSchema(
    inputs = 
    {
        'name': 'seq',
        'doc': '205bp long sequence of PolyA-cut-site',
        'shape': (205, 4),
        'special_type': 'DNASeq'
    },
    targets = 
    {   
        'distal_prop':
        {
            'shape': (1, )
        },
        'doc': 'Predicts proportion of cleavage occuring outside of the specified DNA range',
        'site_props':
        {
            'shape': (205, ),
            'doc':
                'Predicts proportion of cleavage occuring at each position in the specified DNA range. \
                Sum of all site props + distal_prop = 1'
        }
    }
)
dependencies = Dependencies(conda=('python=3.9', 'tensorflow', 'keras>=2.0.4,<3'),
                            conda_channels=('conda-forge', 'bioconda', 'defaults'))
test = KipoiModelTest(
    expect={
        'url': 'https://zenodo.org/record/5511940/files/APARENT.site_probabilities.predictions.hdf5?download=1',
        'md5': '1adb12be84240ffb7d7ca556eeb19e01'
        }
)
doc = 'Predicting the Impact of cis-Regulatory Variation on Alternative Polyadenylation \
Abstract \
Alternative polyadenylation (APA) is a major driver of transcriptome diversity in human cells.'
trained_on = "isoform expression data from over 3 million APA reporters, built by inserting random sequence into 12 distinct 3'UTR contexts."

info = KipoiModelInfo(authors=(Author("Nicholas Bogard"), Author("Johannes Linder")), doc=doc, trained_on=trained_on, 
                        cite_as="https://doi.org/10.1101/300061", contributors=(Author("Shabnam Sadegharmaki", "shabnamsadegh"), 
                        Author("Ziga Avsec", "avsecz"), Author("Muhammed Hasan Çelik", "MuhammedHasan"), Author("Florian R. Hölzlwimmer", "hoeze")))

description = KipoiModelDescription(args=args, schema=schema, info = info, defined_as='model.APARENTModel', dependencies=dependencies, test=test)