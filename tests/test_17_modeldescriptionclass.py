import pytest

from kipoi.kipoimodeldescription import KipoiModelDescription, KipoiModelSchema, Dependencies, KipoiModelTest, KipoiModelInfo, Author

@pytest.fixture
def model_parameters():
    args = {
        'weights': {
            'md5': '4878981d84499eb575abd0f3b45570d3',
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
                'shape': (1, ),
                'doc': 'Predicts proportion of cleavage occuring outside of the specified DNA range',
            },
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
 
    return (args, schema, info, dependencies, test)

def test_modeldescription_missing_modeltype(model_parameters):
    with pytest.raises(ValueError):
        mdc = KipoiModelDescription(args=model_parameters[0], schema=model_parameters[1], 
                            info = model_parameters[2], defined_as='', 
                            dependencies=model_parameters[3], test=model_parameters[4])


def test_basic_modeldescription_class(model_parameters):
    mdc = KipoiModelDescription(args=model_parameters[0], schema=model_parameters[1], 
                            info = model_parameters[2], defined_as='model.APARENTModel', 
                            dependencies=model_parameters[3], test=model_parameters[4])
    assert mdc.args['weights'].md5 == '4878981d84499eb575abd0f3b45570d3'
    assert mdc.schema.inputs.shape[0] == 205
    assert mdc.schema.targets['distal_prop'].shape[0] == 1
    assert mdc.dependencies.conda== ['python=3.9', 'tensorflow', 'keras>=2.0.4,<3']
    assert mdc.dependencies.pip == []
    assert mdc.test.expect['md5'] == '1adb12be84240ffb7d7ca556eeb19e01'
    assert mdc.default_dataloader == '.'
    assert len(mdc.info.authors) == 2
    assert mdc.info.contributors[0].github == "shabnamsadegh"
    assert mdc.writers == {}