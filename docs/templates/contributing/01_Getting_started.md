## Contributing models - Getting started

Kipoi stores models (descriptions, parameter files, dataloader code, ...) as folders in the 
[kipoi/models](https://github.com/kipoi/models) github repository. The minimum requirement for a model is that a 
[`model.yaml`](./02_Writing_model.yaml.md) file is available in the model folder, which defines the type of the model, 
file paths / URLs, the dataloader, description, software dependencies, etc.

We have compiled some of the standard use-cases of model contribution here. Please specify:



<script>
// Definition of dynamic content
var model_class = {"keras": "kipoi.model.KerasModel",
                   "tensorflow": "kipoi.model.TensorFlowModel",
                   "pytorch": "kipoi.model.PyTorchModel",
                   "scikitlearn": "kipoi.model.SklearnModel",
                   "other": "my_model.MyModel # MyModel class defined in my_model.py"};

var model_args = {"keras": `args: # arguments of kipoi.model.KerasModel
    arch:
        url: https://zenodo.org/path/to/my/architecture/file
        md5: 1234567890abc
    weights:
        url: https://zenodo.org/path/to/my/model/weights.h5
        md5: 1234567890abc`,
"tensorflow": `args: # arguments of kipoi.model.TensorFlowModel
  input_nodes: "inputs"
  target_nodes: "preds"
  checkpoint_path: 
      url: https://zenodo.org/path/to/my/model.tf
      md5: 1234567890abc`,
"pytorch": `args: # arguments of kipoi.model.PyTorchModel
    module_class: my_model.DummyModel # DummyModel defined in ./my_model.py
    module_kwargs: # Optional kwargs for the DummyModel initialisation
      x: 1
      y: 2
      z: 3
    weights: # Path to the file containing the state_dict
        url: https://zenodo.org/path/to/my/model/weights.pth
        md5: 1234567890abc`,
"scikitlearn": `args: # arguments of kipoi.model.SklearnModel
  pkl_file: 
      url: https://zenodo.org/path/to/my/model.pkl
      md5: 1234567890abc
  predict_method: predict_proba`,
"other":`args: # Optional. Arguments to be passed to the model initialisation.
  file_path: 
      url: https://zenodo.org/path/to/my/model.pkl
      md5: 1234567890abc
  my_param: 42`,
};
var model_template_args = {"keras": `args: # arguments of kipoi.model.KerasModel
    arch:
        url: {{ model_arch_url }} # refers to models.tsv
        md5: {{ model_arch_md5 }}
    weights:
        url: {{ model_weights_url }}
        md5: {{ model_weights_md5 }}`,
"tensorflow": `args: # arguments of kipoi.model.TensorFlowModel
  input_nodes: "inputs"
  target_nodes: "preds"
  checkpoint_path: 
      url: {{ model_checkpoint_url }} # refers to models.tsv
      md5: {{ model_checkpoint_md5 }}`,
"pytorch": `args: # arguments of kipoi.model.PyTorchModel
    module_class: my_model.DummyModel # DummyModel defined in my_model.py
    module_kwargs: # Optional kwargs for the DummyModel initialisation
      x: 1
      y: 2
      z: 3
    weights: # Path to the file containing the state_dict
        url: {{ model_weights_url }} # refers to models.tsv
        md5: {{ model_weights_md5 }}`,
"scikitlearn": `args: # arguments of kipoi.model.SklearnModel
  pkl_file: 
      url: {{ model_pkl_url }} # refers to models.tsv
      md5: {{ model_pkl_md5 }}
  predict_method: predict_proba`,
"other":`args: # Optional. Arguments to be passed to the model initialisation.
  file_path: 
      url: {{ model_file_url }} # refers to models.tsv
      md5: {{ model_file_md5 }}
  my_param: 42`,
}

var models_tsv = {"keras": `model\tmodel_arch_url\tmodel_arch_md5\tmodel_weights_url\tmodel_weights_md5
my_model_1\thttps://zenodo.org/path/to/my/architecture/file1\t1234567890abc\thttps://zenodo.org/path/to/my/model/weights1.h5\t1234567890abc
my_model_2\thttps://zenodo.org/path/to/my/architecture/file2\t1234567890abc\thttps://zenodo.org/path/to/my/model/weights2.h5\t1234567890abc`,
"tensorflow": `model\tmodel_checkpoint_url\tmodel_checkpoint_md5
my_model_1\thttps://zenodo.org/path/to/my/model1.tf\t1234567890abc
my_model_2\thttps://zenodo.org/path/to/my/model2.tf\t1234567890abc`,
"pytorch": `model\tmodel_weights_url\tmodel_weights_md5
my_model_1\thttps://zenodo.org/path/to/my/model/weights1.pth\t1234567890abc
my_model_2\thttps://zenodo.org/path/to/my/model/weights2.pth\t1234567890abc`,
"scikitlearn": `model\tmodel_pkl_url\tmodel_pkl_md5
my_model_1\thttps://zenodo.org/path/to/my/model1.pkl\t1234567890abc
my_model_2\thttps://zenodo.org/path/to/my/model2.pkl\t1234567890abc`,
"other":`model\tmodel_file_url\tmodel_file_md5
my_model_1\thttps://zenodo.org/path/to/my/model1.pkl\t1234567890abc
my_model_2\thttps://zenodo.org/path/to/my/model2.pkl\t1234567890abc`,
};

var model_yaml_dl_entry ={
    "dna":`
    defined_as: kipoiseq.dataloaders.SeqIntervalDl

    default_args: # Optional arguments to the SeqIntervalDl dataloader
        # See also https://kipoi.org/kipoiseq/dataloaders/#seqintervaldl 
        auto_resize_len: 100 # Automatically resize sequence intervals
        alphabet_axis: 1
        dummy_axis: 2 # Add a dummy axis. Omit in order not to create dummy_axis.
        alphabet: "ACGT" # Order of letters in 1-hot encoding
        ignore_targets: False # if True, dont return any target variables`,
    "dnaAdditional":`. #Refer to dataloader.yaml in the same folder as this file.`,
    "splicing":`
    defined_as: kipoiseq.dataloaders.MMSpliceDl
    default_args: # Optional arguments to the MMSpliceDl dataloader
        intron5prime_len: 100 # 5' intronic sequence length to take.
        intron3prime_len: 100 # 3' intronic sequence length to take.`
};

var model_yaml = `defined_as: {{ model_class }}
{{ model_args }}

default_dataloader: {{ model_yaml_dl_entry }}

info: # General information about the model
    authors: 
        - name: Your Name
          github: your_github_username
          email: your_email@host.org
    doc: Model predicting X
    cite_as: https://doi.org:/... # preferably a doi url to the paper
    trained_on: Dataset Y. held-out chromosomes chr8, chr9 and chr22.
    license: MIT # Software License - if not set defaults to MIT
    # You can also specify the license in the LICENSE file

dependencies:
    conda: # install via conda
      - python
      - h5py
      - pip
    pip:   # install via pip
      - keras&gt;=2.0.4
      - tensorflow&gt;=1.0

schema:  # Model schema. The schema defintion is essential for kipoi plug-ins to work.
    inputs:  # input = single numpy array
        shape: (100,4)  # array shape of a single sample (omitting the batch dimension)
        doc: input feature description

    # inputs:  # input = dictionary of fields
    #   seq:
    #     shape: (100,4)
    #     doc: input feature description
    #   other_track:
    #     shape: (50,)
    #     doc: input feature description
    targets:
        shape: (3,)
        doc: model prediction description
`;

var model_py = `from kipoi.model import BaseModel

class MyModel(BaseModel): # Implement your Kipoi model
    def __init__(self, file_path, my_param):
        ...
        self.model = load_model_parameters(file_path)

    # Execute model prediction for input data
    def predict_on_batch(self, x): # The bare minimum that has to be defined
        return self.model.predict(x)`;

var dataloader_yaml = `defined_as: dataloader.MyDataset # MyDataset impolemented in dataloader.py
args: # MyDataset.__init__ argument description
    features_file:
        doc: intervals_file: bed3 file containing intervals
        # Test file URL's
        example: 
            url: https://raw.githubusercontent.com/../intervals_51bp.tsv
            md5: a76e47b3df87fd514860cf27fdc10eb4
    targets_file:
        doc: Reference genome FASTA file path.
        example:
            url: https://raw.githubusercontent.com/../hg38_chr22_32000000_32300000.fa
            md5: 01320157a250a3d2eea63e89ecf79eba
    ignore_targets:
        doc: if True, don't return any target variables
        optional: True  # if not present, the "targets" will not be present

info:
    authors: 
        - name: Your Name
          github: your_github_account
          email: your_email@host.org
    doc: Data-loader returning one-hot encoded sequences given genome intervals

dependencies:
    conda:
      - python
      - bioconda::pybedtools
      - bioconda::pysam
      - bioconda::pyfaidx
      - numpy
      - pandas
      - pip
    pip:
      - kipoiseq

output_schema: # Define the dataloader output schema according to the returned values
    inputs:
        seq:
            shape: (100, 4)
            doc: One-hot encoded DNA sequence
        other_track:
            shape: (50,)
            doc: dummy track
    targets:
        shape: (None,)
        doc: (optional) values following the bed-entry
    metadata:  # additional information about the samples
        ranges:
            type: GenomicRanges
            doc: Ranges describing inputs.seq`;

var dataloader_py = `from __future__ import absolute_import, division, print_function
import numpy as np
from kipoi.data import Dataset
from kipoi.metadata import GenomicRanges
from kipoiseq.dataloaders.sequence import BedDataset
from kipoiseq.extractors import FastaStringExtractor
from kipoiseq.transforms import OneHot


class MyDataset(Dataset):
    """Example re-implementation of kipoiseq.dataloaders.SeqIntervalDl

    Args:
        intervals_file: bed3 file containing intervals
        fasta_file: file path; Genome sequence
    """

    def __init__(self, intervals_file, fasta_file, ignore_targets=True):
        self.bt = BedDataset(intervals_file,
                             bed_columns=3,
                             ignore_targets=ignore_targets)
        self.fasta_file = fasta_file
        self.fasta_extractor = None
        self.transform = OneHot()  # one-hot encode DNA sequence

    def __len__(self):
        return len(self.bt)

    def __getitem__(self, idx):
        if self.fasta_extractor is None:
            self.fasta_extractor = FastaStringExtractor(self.fasta_file)

        # get the intervals
        interval, targets = self.bed[idx]

        # resize to 100bp
        interval = resize_interval(interval, 100, anchor='center')

        # extract the sequence
        seq = self.fasta_extractors.extract(interval)

        # one-hot encode the sequence
        seq_onehot = self.transform(seq)

        return {
            "inputs": {
               "seq": seq_onehot,
               "other_track": np.ones((50, ))
             },
            "targets": targets,   # (optional field)
            "metadata": {
                "ranges": GenomicRanges.from_interval(interval)
            }
        }`;
</script>


<div class="container">
    <p> 
        <ul>
            <li>which input data your model requires:
                <select id="sel_inp" onchange="refresh_info()">
                    <option value="None" selected>Select...</option>
                    <option value="dna">DNA sequence (one-hot encoded or string)</option>
                    <option value="dnaAdditional">DNA sequence with additional tracks</option>
                    <option value="splicing">DNA sequence splicing model</option>
                    <option value="otherInput">Other model input</option>
                </select>
            </li>
            <li>
                in which framework your model is implemented:
                <select id="sel_fw" onchange="refresh_info()">
                    <option value="None" selected>Select...</option>
                    <option value="keras">Keras</option>
                    <option value="tensorflow">TensorFlow</option>
                    <option value="pytorch">PyTorch</option>
                    <option value="scikitlearn">Sci-Kit learn</option>
                    <option value="other">other</option>
                </select>
            </li>
            <li>
                whether you want to contribute a:
                <select id="sel_mg" onchange="refresh_info()">
                    <option value="None" selected>Select...</option>
                    <option value="single">single model</option>
                    <option value="setSim">set of highly similar models (say models for different TFs)</option>
                    <option value="setDiff">set of models that logically belong together, but may not be very similar</option>
                </select>
            </li>
        </ul>
    </p>
</div>


<style>
.cond:{
    visibility: hidden;
}
.hidden:{
    visibility: hidden;
}
</style>

<!--- BEGIN extra imports for yaml display etc. --->
<script src="../../js/jquery-2.1.1.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/js/bootstrap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mustache.js/3.0.1/mustache.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/1.5.16/clipboard.min.js"></script>
<link rel="stylesheet" type="text/css" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css">
<!--- END extra imports for yaml display etc. --->

<p></p>
<p></p>
<!-- YAML goes here. Tabs for the different yamls and python files -->


<div class="cond" style="display: none;">
    <h3 id="preparation">Preparation</h3>
</div>

<div class="cond forking" style="display: none;">
    <p>Before you start, make sure you have installed <code>kipoi</code>.</p>
</div>


<div class="cond anyExpl" style="display: none;">
    <h3 id="setting-up-your-model">Setting up your model</h3>
</div>
    
<div class="cond single setSim" style="display: none;">
    <p>For this example let's assume the model you want to submit is called <code>MyModel</code>. To submit your model
    you will have create the folder <code>MyModel</code> in you Kipoi model folder (default:
    <code>~/.kipoi/models</code>). In this folder you will have to create the following file(s):</p>
</div>

<div class="cond setDiff" style="display: none;">

    <p>If you have trained multiple models that logically belong into one model-group as they are similar in function, but 
    they individually require different preprocessing code then you are right here. To submit your model you will have to:</p>
    <ul>
        <li>Create a new local folder named after your model, e.g.: <br /><code>mkdir MyModel</code><br /> and within this folder create a folder
        structure so that every individual trained model has its own folder. Every folder that contains a <code>model.yaml</code> is then
        interpreted as an individual model by Kipoi.</li>
        <li>To make this clearer take a look at how <code>FactorNet</code> is structured: 
        <a href="https://github.com/kipoi/models/tree/master/FactorNet">FactorNet</a>. If you have files that are re-used in multiple 
        models you can use symbolic links (<code>ln -s</code>) relative within the folder structure of your model group.</li>
        <li>For your selection the following files have to exist in every sub-folder that should act as an individual model:</li>
    </ul>

</div>


<!--- this should be the copy-to-clipboard button that was placed in the tab-pane
<button type="button" class="btn btn-default clipboard-btn hidden" data-clipboard-target="#model_yaml_raw_code">Copy to clipboard</button>
<input class="hidden" id="model_yaml_raw_code"></input>
--->

<div class="cond anyExpl" style="display: none;">
    <div id="wrapper">
        <!-- Nav tabs -->
        <ul id="codes" class="nav nav-tabs" role="tablist">
          <li role="presentation" class="cond single setDiff" id="top-tab-model_yaml"><a href="#tab-model_yaml" role="tab" data-toggle="tab">model.yaml</a></li>
          <li role="presentation" class="cond setSim" id="top-tab-model-template_yaml"><a href="#tab-model-template_yaml" role="tab" data-toggle="tab">model-template.yaml</a></li>
          <li role="presentation" class="cond other"><a href="#tab-model_py" role="tab" data-toggle="tab">model.py</a></li>
          <li role="presentation" class="cond setSim"><a href="#tab-models_tsv" role="tab" data-toggle="tab">models.tsv</a></li>
          <li role="presentation" class="cond dnaAdditional otherInput"><a href="#tab-dataloader_yaml" role="tab" data-toggle="tab">dataloader.yaml</a></li>
          <li role="presentation" class="cond dnaAdditional otherInput"><a href="#tab-dataloader_py" role="tab" data-toggle="tab">dataloader.py</a></li>
        </ul>
    
        <!-- Tab panes -->
        <div class="tab-content">
            <div role="tabpanel" class="tab-pane" id="tab-model_yaml">
                <pre><code class="yaml hljs makefile" id="model_yaml_code">
                </code></pre>
            </div>
            <div role="tabpanel" class="tab-pane" id="tab-model-template_yaml">
                <pre><code class="yaml hljs makefile" id="model-template_yaml_code">
                </code></pre>
            </div>
            <div role="tabpanel" class="tab-pane" id="tab-model_py">
                <pre><code class="python hljs" id="model_py_code"></code></pre>
            </div>
            <div role="tabpanel" class="tab-pane" id="tab-models_tsv">
                <pre><code class="yaml hljs makefile" id="models_tsv_code"></code></pre>
            </div>
            <div role="tabpanel" class="tab-pane" id="tab-dataloader_yaml">
                <pre><code class="yaml hljs makefile" id="dataloader_yaml_code"></code></pre>
            </div>
            <div role="tabpanel" class="tab-pane" id="tab-dataloader_py">
                <pre><code class="python hljs" id="dataloader_py_code"></code></pre>
            </div>
        </div>
    </div>
</div>

<div class="cond" style="display: none;">
    <p>For this example let's assume the model you want to submit is called <code>MyModel</code>. To submit your model you will have to:</p>
    <ul>
        <li>Create a new local folder named like your model, e.g.: <code>mkdir MyModel</code></li>
        <li>In the <code>MyModel</code> folder you will have to crate a <code>model.yaml</code> file:
            The <code>model.yaml</code> files acts as a configuration file for Kipoi. For an example take a look at 
            <a href="https://github.com/kipoi/models/blob/master/Divergent421/model.yaml">Divergent421/model.yaml</a>.</li>
    </ul>
</div>

<div class="cond" style="display: none;">

    <p>For this example let's assume you have trained one model architecture on multiple similar datasets and can use the 
     same preprocessing code for all models. Let's assume you want to call the 
    model-group <code>MyModel</code>. To submit your model you will have to:</p>
    <ul>
        <li>Create a new local folder named after your model, e.g.: <code>mkdir MyModel</code></li>
        <li>In the <code>MyModel</code> folder you will have to crate a <code>model-template.yaml</code> file:
            The <code>model-template.yaml</code> files acts as a configuration file for Kipoi. For an example take a look at 
            <a href="https://github.com/kipoi/models/blob/master/CpGenie/model-template.yaml">CpGenie/model-template.yaml</a>.</li>
        <li>As you can see instead of putting urls and parameters directly in the <code>.yaml</code> file you need to put 
        <code>{{ parameter_name }}</code> in the yaml file. The values are then automatically loaded from a <code>tab</code>-delimited
        file called <code>models.tsv</code> that you also have to provide. For the previous example this would be: 
        <a href="https://github.com/kipoi/models/blob/master/CpGenie/models.tsv">CpGenie/models.tsv</a>. Using kipoi those models are
        then accessible by the model group name and the model name defined in the <code>models.tsv</code>. Model names may contain <code>/</code>s.</li>
    </ul>


</div>


<div class="cond" style="display: none;">

    <ul>
        <li>In the model definition yaml file you see the <code>defined_as</code> keyword: Since your model is a Keras model, set it to
         <code>kipoi.model.KerasModel</code>.</li>
        <li>In the model definition yaml file you see the <code>args</code> keyword, which can be set the following way: 
        <a href="../02_Writing_model.yaml/#kipoimodelkerasmodel-models">KerasModel definition</a></li>
    </ul>


</div>

<div class="cond" style="display: none;">

    <ul>
        <li>In the model definition yaml file you see the <code>defined_as</code> keyword: Since your model is a TensorFlow model, set it to
         <code>kipoi.model.TensorFlowModel</code>.</li>
        <li>In the model definition yaml file you see the <code>args</code> keyword, which can be set the following way: 
        <a href="../02_Writing_model.yaml/#kipoimodeltensorflowmodel-models">TensorFlowModel definition</a></li>
    </ul>


</div>

<div class="cond" style="display: none;">

    <ul>
        <li>In the model definition yaml file you see the <code>defined_as</code> keyword: Since your model is a PyTorch model, set it to
         <code>kipoi.model.PyTorchModel</code>.</li>
        <li>In the model definition yaml file you see the <code>args</code> keyword, which can be set the following way: 
        <a href="../02_Writing_model.yaml/#kipoimodelpytorchmodel-models">PyTorchModel definition</a></li>
    </ul>


</div>

<div class="cond" style="display: none;">

    <ul>
        <li>In the model definition yaml file you see the <code>defined_as</code> keyword: Since your model is a scikit-learn model, set it to
         <code>kipoi.model.SklearnModel</code>.</li>
        <li>In the model definition yaml file you see the <code>args</code> keyword, which can be set the following way: 
        <a href="../02_Writing_model.yaml/#kipoimodelsklearnmodel-models">SklearnModel definition</a></li>
    </ul>


</div>

<div class="cond" style="display: none;">

    <ul>
        <li>Your model is not implemented in <code>Keras</code>, <code>TensorFlow</code>, <code>PyTorch</code>, nor <code>sci-kit learn</code>, so you will have to implement a 
        custom python class inheriting from <code>kipoi.model.Model</code>. In the <code>defined_as</code> keyword of the <code>model.yaml</code> you will then 
        have to refer to your definition by <code>my_model_def.MyModel</code> if the <code>MyModel</code> class is defined in the <code>my_model_def.py</code> 
        that lies in the same folder as <code>model.yaml</code>. For details please see: 
        <a href="../02_Writing_model.yaml/#custom-models">defining custom models in model.yaml</a> and 
        <a href="../05_Writing_model.py">writing a model.py file</a>.</li>
    </ul>

</div>

<div class="cond" style="display: none;">

<ul>
    <li>Now set the software requirements correctly. This happens in the <code>dependencies</code> section of the model 
    <code>.yaml</code> file. As you can see in the example the dependencies are split by <code>conda</code> and <code>pip</code>. Ideally you define the 
    ranges of the versions of packages your model supports - otherwise it may fail at some point in future. If you need 
    to specify a conda channel use the <code>&lt;channel&gt;::&lt;package&gt;</code> notation for conda dependencies.</li>
</ul>

</div>

<div class="cond" style="display: none;">

    <p>As you have seen in the presented example and in the model definition links it is necessary that prior to model 
    contribution you have published all model files (except for python scripts and other configuration files) on 
    <a href="https://zenodo.org/">zenodo</a> or <a href="https://figshare.com/">figshare</a> to ensure functionality and versioning of models.</p>
    <p>If you want to test your model(s) locally before publishing them on <a href="https://zenodo.org/">zenodo</a> or
     <a href="https://figshare.com/">figshare</a> you can replace the pair of <code>url</code> and <code>md5</code> tags in the model definition yaml by the 
    local path on your filesystem, e.g.:</p>
    <pre><code class="yaml hljs makefile"><span class="hljs-section">args:</span>
        arch: path/to/my/arch.json
    </code></pre>

    <p>But keep in mind that local paths are only good for testing and for models that you want to keep only locally.</p>

</div>

<div class="cond dnaAdditional" style="display: none;">

    <h3 id="setting-up-your-dataloader">Setting up your dataloader</h3>

</div>

<div class="cond" style="display: none;">

    <p>Sice your model uses DNA sequence input the <a href="https://github.com/kipoi/kipoiseq">kipoiseq</a> dataloaders are recommended to be used, as shown in 
    the above example model definition <code>.yaml</code> file, which could for example be defined like this:</p>
    <pre><code class="yaml hljs css"><span class="hljs-selector-tag">default_dataloader</span>:
      <span class="hljs-selector-tag">defined_as</span>: <span class="hljs-selector-tag">kipoiseq</span><span class="hljs-selector-class">.dataloaders</span><span class="hljs-selector-class">.SeqIntervalDl</span>
      <span class="hljs-selector-tag">default_args</span>:
        <span class="hljs-selector-tag">auto_resize_len</span>: 1001
        <span class="hljs-selector-tag">alphabet_axis</span>: 0
        <span class="hljs-selector-tag">dummy_axis</span>: 1
    </code></pre>

    <p>To see all the parameters and functions of the off-the-shelf dataloaders please take a look at 
    <a href="https://github.com/kipoi/kipoiseq">kipoiseq</a>.</p>

</div>

<div class="cond dnaAdditional" style="display: none;">

    <p>Since your model uses DNA sequence and additional annotation you have to define your own dataloader function or class. 
    Depending on your use-case you may find some of the data-loader implementations of exiting models in the model zoo 
    helpful. You may find the 
    <a href="https://github.com/kipoi/models/blob/master/rbp_eclip/dataloader.py">rbp_eclip dataloader</a> or one of the 
    <a href="https://github.com/kipoi/models/blob/master/FactorNet/CEBPB/meta_Unique35_DGF/dataloader.py">FactorNet dataloaders</a> 
    relevant. Also consider taking advantage of elements implemented in the <a href="https://github.com/kipoi/kipoiseq">kipoiseq</a> 
    package. For you implementation you have to:</p>
    <ul>
        <li>set <code>default_dataloader: .</code> in the <code>model.yaml</code> file</li>
        <li>write a <code>dataloader.yaml</code> file as defined in <a href="../03_Writing_dataloader.yaml">writing dataloader.yaml</a>. An example is 
        <a href="https://github.com/kipoi/models/blob/master/FactorNet/CEBPB/meta_Unique35_DGF/dataloader.yaml">this one</a>.</li>
        <li>implement the dataloader in a <code>dataloader.py</code> file as defined in 
        <a href="../03_Writing_dataloader.py">writing dataloader.py</a>. An example is 
        <a href="https://github.com/kipoi/models/blob/master/FactorNet/CEBPB/meta_Unique35_DGF/dataloader.py">this one</a>.</li>
        <li>put the <code>dataloader.yaml</code> and the <code>dataloader.py</code> in the same folder as <code>model.yaml</code>.</li>
    </ul>

</div>

<div class="cond otherInput" style="display: none;">

    <p>Since your model uses input other than what is covered by the default data-loaders you have to define your own 
    dataloader function or class. 
    Depending on your use-case you may find some of the data-loader implementations of exiting models in the model zoo 
    helpful. You may find the 
    <a href="https://github.com/kipoi/models/blob/master/rbp_eclip/dataloader.py">rbp_eclip dataloader</a> or one of the 
    <a href="https://github.com/kipoi/models/blob/master/FactorNet/CEBPB/meta_Unique35_DGF/dataloader.py">FactorNet dataloaders</a> 
    relevant. Also consider taking advantage of elements implemented in the <a href="https://github.com/kipoi/kipoiseq">kipoiseq</a> 
    package. For you implementation you have to:</p>
    <ul>
        <li>set <code>default_dataloader: .</code> in the <code>model.yaml</code> file</li>
        <li>write a <code>dataloader.yaml</code> file as defined in <a href="../03_Writing_dataloader.yaml">writing dataloader.yaml</a>. An example is 
        <a href="https://github.com/kipoi/models/blob/master/FactorNet/CEBPB/meta_Unique35_DGF/dataloader.yaml">this one</a>.</li>
        <li>implement the dataloader in a <code>dataloader.py</code> file as defined in 
        <a href="../03_Writing_dataloader.py">writing dataloader.py</a>. An example is 
        <a href="https://github.com/kipoi/models/blob/master/FactorNet/CEBPB/meta_Unique35_DGF/dataloader.py">this one</a>.</li>
        <li>put the <code>dataloader.yaml</code> and the <code>dataloader.py</code> in the same folder as <code>model.yaml</code>.</li>
    </ul>

</div>

<div class="cond splicing" style="display: none;">

    <p>Since your model is specialised in predicting properties of splice sites you are encouraged to take a look at the 
    dataloaders implemented for the kipoi models tagged as <code>RNA splicing</code> models, such as 
    <a href="https://github.com/kipoi/models/blob/master/HAL/dataloader.py">HAL</a>, 
    <a href="https://github.com/kipoi/models/blob/master/labranchor/dataloader.py">labranchor</a>, or 
    <a href="https://github.com/kipoi/models/tree/master/MMSplice">MMSplice</a>.
     If the MMSplice dataloader in the above example does not fit your needs, you have to:</p>
    <ul>
        <li>set <code>default_dataloader: .</code> in the <code>model.yaml</code> file</li>
        <li>write a <code>dataloader.yaml</code> file as defined in <a href="../03_Writing_dataloader.yaml">writing dataloader.yaml</a>.</li>
        <li>implement the dataloader in a <code>dataloader.py</code> file as defined in 
        <a href="../03_Writing_dataloader.py">writing dataloader.py</a>.</li>
        <li>put the <code>dataloader.yaml</code> and the <code>dataloader.py</code> in the same folder as <code>model.yaml</code>.</li>
    </ul>

</div>

<div class="cond" style="display: none;">

    <h3 id="info-and-model-schema">Info and model schema</h3>
    
    <p> Please update the model description, the authors and the data it the model was trained in the <code>info</code> 
    section of the model <code>.yaml</code> file. Please explain explicitly what your model does etc. Think what you 
    would want to know if you didn't know anything about the model.</p>
    
    <p>Now fillout the model schema (<code>schema</code> tag) as explained here: 
    <a href="../#02_Writing_model.yaml/#schema">model schema</a>.</p>

</div>


<div class="cond anyExpl" style="display: none;">

    <h3 id="license">License</h3>

    <p>Please make sure that the license that is defined in the <code>license:</code> tag in the yaml file is correct.
    Also only contribute models for which you have the rights to do so and only contribute models that permit 
    redistribution.</p>

</div>

<div class="cond single" style="display: none;">

    <h3 id="testing">Testing</h3>

    <p> Now it is time to test your model. If you are in the model directory run the command:</p>
    <pre><code class="hljs bash">kipoi test .</code></pre>
    <p>in your model folder to test 
    whether the general setup is correct. When this was successful run </p>
     <pre><code class="hljs bash">kipoi test-source dir --all</code></pre>
    <p>to test whether all the software dependencies of the model are setup correctly and the automated tests will 
    pass.</p>

</div>

<div class="cond setSim setDiff" style="display: none;">

    <h3 id="testing">Testing</h3>

    <p> Now it is time to test your models. For the following let's assume your model group is called 
    <code>MyModel</code> and your have two models in the group, which are <code>MyModel/ModelA</code> and 
    <code>MyModel/ModelB</code> then you should should make sure you are in the <code>MyModel</code> folder and 
    run the commands </p>
    <pre><code class="hljs bash">kipoi test ./ModelA</code></pre>
    <p> and </p>
    <pre><code class="hljs bash">kipoi test ./ModelB</code></pre>
    <p>. When this was successful 
    run </p>
    <pre><code class="hljs bash">kipoi test-source dir --all</code></pre>
    <p> to test whether all the software dependencies of the model and dataloader are setup correctly.</p>

</div>


<div class="cond forking" style="display: none;">

    <h3 id="forking-and-submitting">Forking and submitting</h3>

    <ul>
    <li>Make sure your model repository is up to date: <ul>
    <li><code class="hljs bash">git pull</code></li>
    </ul>
    </li>
    <li>Commit your changes<ul>
    <li><code class="hljs bash">git add MyModel/</code></li>
    <li><code class="hljs bash">git commit -m "Added &lt;MyModel&gt;"</code></li>
    </ul>
    </li>
    <li><a href="https://guides.github.com/activities/forking/">Fork</a> the <a href="https://github.com/kipoi/models">https://github.com/kipoi/models</a> repo on github (click on 
    the Fork button)</li>
    <li>Add your fork as a git remote to <code>~/.kipoi/models</code><ul>
    <li><code class="hljs bash">git remote add fork https://github.com/&lt;username&gt;/models.git</code></li>
    </ul>
    </li>
    <li>Push to your fork<ul>
    <li><code class="hljs bash">git push fork master</code></li>
    </ul>
    </li>
    <li>Submit a pull-request<ul>
    <li>On github click the <a href="https://help.github.com/articles/creating-a-pull-request/">New pull request</a> button on your 
    github fork - <code>https://github.com/&lt;username&gt;/models&gt;</code></li>
    </ul>
    </li>
    </ul>
</div>


<script type="text/javascript">

get_model_yaml_code = function(){
    var sel_inp = $('#sel_inp').val();
    var sel_fw = $('#sel_fw').val();
    return Mustache.render(model_yaml, {model_class: model_class[sel_fw], model_args: model_args[sel_fw], model_yaml_dl_entry: model_yaml_dl_entry[sel_inp]});
}

get_model_template_yaml_code = function(){
    var sel_inp = $('#sel_inp').val();
    var sel_fw = $('#sel_fw').val();
    return Mustache.render(model_yaml, {model_class: model_class[sel_fw], model_args: model_template_args[sel_fw], model_yaml_dl_entry: model_yaml_dl_entry[sel_inp]});
}

get_models_tsv_code = function(){
    var sel_fw = $('#sel_fw').val();
    return models_tsv[sel_fw];
}

get_model_py_code = function(){
    return model_py;
}

get_dataloader_py_code = function(){
    return dataloader_py;
}

get_dataloader_yaml_code = function(){
    return dataloader_yaml;
}

function copyToClipboard(text){
    //https://stackoverflow.com/questions/33855641/copy-output-of-a-javascript-variable-to-the-clipboard
    var dummy = document.createElement("input");
    document.body.appendChild(dummy);
    dummy.setAttribute('value', unescape(text));
    dummy.select();
    document.execCommand("copy");
    document.body.removeChild(dummy);
}


insert_code_data = function(){
    var sel_inp = $('#sel_inp').val();
    var sel_fw = $('#sel_fw').val();
    var sel_mg = $('#sel_mg').val();
    //$("#model_yaml_raw_code").val(unescape(get_model_yaml_code())); // for copying to clipboard
    $("#model_yaml_code").html(get_model_yaml_code());
    $("#model-template_yaml_code").html(get_model_template_yaml_code());
    $("#models_tsv_code").html(get_models_tsv_code());
    $("#model_py_code").html(get_model_py_code());
    $("#dataloader_py_code").html(get_dataloader_py_code());
    $("#dataloader_yaml_code").html(get_dataloader_yaml_code());    
    $('.tab-pane').each(function(i, block) {
      hljs.highlightBlock(block);
    });
}


refresh_info = function(){
    $('.cond').hide();
    var sel_inp = $('#sel_inp').val();
    var sel_fw = $('#sel_fw').val();
    var sel_mg = $('#sel_mg').val();
    // deactivate the code tabs
    $(".nav-tabs").children().removeClass("active")
    $(".tab-pane").removeClass("active")
    if (($.inArray(sel_inp, ['dna', 'dnaAdditional', 'splicing', 'otherInput'])>-1) && ($.inArray(sel_fw, ['keras', 'tensorflow', 'pytorch', 'scikitlearn', 'other'])>-1) && ($.inArray(sel_mg, ['single', 'setSim', 'setDiff'])>-1)){
        insert_code_data();
        if (sel_mg == "setSim"){
            //assign active class to top-tab-model-template_yaml and tab-model-template_yaml
            $("#top-tab-model-template_yaml").addClass("active");
            $("#tab-model-template_yaml").addClass("active");
        } else {
            //assign active class to top-tab-model_yaml and tab-model_yaml
            $("#top-tab-model_yaml").addClass("active");
            $("#tab-model_yaml").addClass("active");
        }
        $(".anyExpl").show();
        $(".forking").show();
        $("."+sel_inp).show();
        $("."+sel_fw).show();
        $("."+sel_mg).show();
    }
}
refresh_info();


</script>
