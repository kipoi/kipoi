"""
## Setup ~/.kipoi/config.yaml

echo "Configure Kipoi"
mkdir -p ~/.kipoi
echo "
model_sources:
  kipoi:
    type: git
    remote_url: git@github.com:kipoi/models.git
    local_path: /data/kipoi/models/
    auto_update: False
" > ~/.kipoi/config.yaml

# clone the models
git clone git@github.com:kipoi/models.git -o /data/kipoi/models

pip install kipoi

# create all the environments
kipoi env create shared/envs/kipoi-py3-keras2
kipoi env create shared/envs/kipoi-py3-keras1.2

## Running models
kipoi_cmd=$(kipoi env get_cli {wildcards.model})

$kipoi_cmd veff score_variants \
    {wildcards.model} \
    --dataloader_args='{{"fasta_file": "{input.fasta_file}"}}' \
    -i {input.vcf} \
    -n {params.workers} \
    --batch_size={params.batch_size} \
    {params.output_flag} {output.f} \
    -s {params.score} \
    --std_var_id
"""
import os
import kipoi
from collections import OrderedDict
from kipoi_utils.utils import cd
from kipoi.specs import ArraySchema
import numpy as np
from joblib import Parallel, delayed


def get_output_schema(targets):
    """Get the output schema as a json-able object
    """
    DTYPE = 'float32'
    try:
        if isinstance(targets, ArraySchema):
            if targets.shape == (1,) and targets.column_labels is None:
                return DTYPE
            if len(targets.shape) == 1:
                if targets.column_labels is not None:
                    # schema has column-names. Use a dictionary as output

                    # NODE: I added i in order to support non-unique column names (which
                    # don't actually make sense but still occur)
                    return {cl + "_" + str(i): DTYPE for i, cl in enumerate(targets.column_labels)}

            # fully general array
            return np.full(targets.shape, DTYPE).tolist()
        elif isinstance(targets, dict):
            return {k: get_output_schema(v) for k, v in targets.items()}
        elif isinstance(targets, list):
            return [get_output_schema(v) for v in targets]
        else:
            raise ValueError("Expected ArraySchema, list or dict")
    except:
        print(targets)
        return None
        # import ipdb
        # ipdb.set_trace()


def get_dataloader_descr(model_name, source='kipoi'):
    """Not yet nicely integrated with Kipoi

    Args:
      model_name: model name as a string

    Returns:
      (model output schema, list of required files)
    """
    dl_skip_arguments = {
        "kipoiseq.dataloaders.SeqIntervalDl": ['alphabet_axis', 'dummy_axis', 'alphabet', 'dtype']
    }
    md = kipoi.get_model_descr(model_name)
    src = kipoi.get_source(source)

    # get dataloader
    if isinstance(md.default_dataloader, str):
        dataloader = kipoi.get_dataloader_descr(os.path.join(model_name, md.default_dataloader),
                                                source=source)
        dataloader_name = md.default_dataloader
        dataloader_args = dataloader.args
    else:
        with cd(src.get_model_dir(model_name)):
            dataloader = md.default_dataloader.get()
        dataloader_name = md.default_dataloader.defined_as
        dataloader_args = OrderedDict([(k, v)
                                       for k, v in dataloader.args.items()
                                       if k not in list(md.default_dataloader.default_args) +
                                       dl_skip_arguments.get(dataloader_name, [])])

        if md.default_dataloader.defined_as == 'kipoiseq.dataloaders.SeqIntervalDl':
            # HACK - cleanup some values for SeqIntervalDl
            if md.default_dataloader.default_args.get("ignore_targets", False):
                dataloader_args.pop('label_dtype', None)

    required_files = []
    if 'fasta_file' in dataloader.args:
        required_files.append("fasta_file")
    if 'gtf_file' in dataloader.args:
        required_files.append("gtf_file")

    return get_output_schema(md.schema.targets), required_files


def test_get_dataloader_descr():
    assert get_dataloader_descr("HAL", source='kipoi') == ('float32', ['fasta_file', 'gtf_file'])
    schema, files = get_dataloader_descr("DeepSEA/variantEffects", source='kipoi')
    assert files == ['fasta_file']
    assert isinstance(schema, dict)
    assert len(schema) == 919

    schema, files = get_dataloader_descr("Basset", source='kipoi')
    assert files == ['fasta_file']
    assert isinstance(schema, dict)
    assert len(schema) == 164

    schema, files = get_dataloader_descr("DeepBind/Homo_sapiens/TF/D00328.018_ChIP-seq_CTCF", source='kipoi')
    assert schema, files == ('float32', ['fasta_file'])


df = kipoi.list_models()


# get all models supporting variant effect prediction
dfv = df[df.veff_score_variants]

# not sure why this wasn't kicked out
dfv = dfv[~dfv.model.str.startswith("extended_coda")]

# DeepCpG_DNA has variable length output which I don't know how to handle
dfv = dfv[~dfv.model.str.startswith("DeepCpG_DNA")]

# append the annotation for Hail
dfv['hail_annotation'] = Parallel(4)(delayed(get_dataloader_descr)(m) for m in dfv.model)

dfv.to_pickle("kipoi-veff-models.pkl")

# --------------------------------------------
# read the resulting file
import pandas as pd
df_anno = pd.read_pickle("kipoi-veff-models.pkl")
df_anno.iloc[0].hail_annotation
# tuple containing the output schema and the required files (gtf_file or fasta_file)
