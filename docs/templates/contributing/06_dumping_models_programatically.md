## Contributing multiple very similar models

Consider an example where multiple models were trained, each for a different cell-lines (case for CpGenie). Here is the final folder structure of the contributed model *group* (simplifed from te [CpGenie](https://github.com/kipoi/models/tree/master/CpGenie) model)

```
cell_line_1
├── dataloader.py -> ../template/dataloader.py
├── dataloader.yaml -> ../template/dataloader.yaml
├── example_files -> ../template/example_files
├── model_files
│   └── model.h5
└── model.yaml -> ../template/model.yaml

cell_line_2
├── dataloader.py -> ../template/dataloader.py
├── dataloader.yaml -> ../template/dataloader.yaml
├── example_files -> ../template/example_files
├── model_files
│   └── model.h5
└── model.yaml -> ../template/model.yaml

template
├── dataloader.py
├── dataloader.yaml
├── example_files
│   ├── hg38_chr22.fa
│   ├── hg38_chr22.fa.fai
│   └── intervals.bed
└── model.yaml
Makefile
test_subset.txt
```

## `template/` folder

The `template/` folder should contain all the common files or templates. This directory is ignored when listing models.

### Softlinks

One option to prevent code duplication is to use soft-links. In the simplest case (as shown above), all files except model weights can be shared accross models. When selectively downloading files from git-lfs, Kipoi also considers soft-links and downloads the original files (e.g. when running `kipoi predict my_model/cell_line_1 ...`, the git-lfs files in `my_model/template` will also get downloaded).

**Note** Make sure you are using **relative** soft-links (as shown above). 

```python
# example code-snippet to dump of multiple Keras models
# and to softlink the remaining files

def get_model(cell_line):
    """Returns the Keras model"""
    pass


def write_model(root_path, cell_line):
    """For a particular cell_line:
	- write out the model
	- softlink the other files from `template/`
    """
    model_dir = os.path.join(root_path, cell_line)
    os.makedirs(os.path.join(model_dir, "model_files"), exist_ok=True)
    
    model = get_model(cell_line)
    
    model.save(os.path.join(model_dir, "model_files/model.h5"))
    
    symlink_files = ["model.yaml", 
                     "example_files", 
                     "dataloader.yaml", 
                     "dataloader.py"]
    for f in symlink_files:
        os.symlink(os.path.join(root_path, "template", f),
                   os.path.join(model_dir, f))


for cell_line in all_cell_lines:
    write_model("my_model_path", cell_line)
```


### Jinja templating

Another option is to use template engines. Template engines are heavily used in web-development to dynamically generate html files. One of the most popular template engines is [jinja](http://jinja.pocoo.org/). Template engines offer more flexibility over softlinks. With softlinks you can only re-use the whole file, while with templating you can choose which pieces of the file are shared and which ones are specific to each model.

```yaml
# template_model.yaml
type: keras
args:
    weights: model_files/model.h5
...
info:
  trained_on: DNase-seq of {{cell_line}} cell line
  ...
schema:
	...
	targets:
	  name: output
	  doc: DNA accessibility in {{cell_line}} cell line
```


```python
# Script to generate <cell line>/model.yaml from template_model.yaml
import os
from jinja2 import Template

def render_template(template_path, output_path, context, mkdir=False):
    """Render template with jinja

    Args:
      template_path: path to the jinja template
      output_path: path where to write the rendered template
      context: Dictionary containing context variable
    """
	if mkdir:
	    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(template_path, "r") as f:
        template = Template(f.read())
    out = template.render(**context)
    with open(output_path, "w") as f:
        f.write(out)


def write_model_yaml(root_path, cell_line):
    """For a particular cell_line:
	- Generate `{cell_line}/model.yaml`
    """

	render_template(os.path.join(root_path, "template", "template_model.yaml"),
	                os.path.join(root_dir, cell_line, "model.yaml"),
					context={"cell_line": cell_line},
					mkdir=True)
```

### Importing common functions, classes

In case the dataloaders or custom models vary between models and we want to re-use python code, we can import objects from modules in the `template/` directory:


```python
import os
import inspect

# Get the directory of this python file
filename = inspect.getframeinfo(inspect.currentframe()).filename
this_path = os.path.dirname(os.path.abspath(filename))

# attach template to pythonpath
import sys
sys.path.append(os.path.join(this_path, "../template"))

from model_template import TemplateModel

class SpecificModel(TemplateModel):
    def __init__(self):
        super(SpecificModel, self).__init__(arg1="value")
```


## `test_subset.txt` - Testing only some models

Since many models are essentially the same, the automatic tests should only test one or few models. To specify which models to test,
write the `test_subset.txt` file in the same directory level as the `template/` folder and list the models you want to test.

Examples:

`CpGenie/test_subset.txt`: 
```
GM19239_ENCSR000DGH
merged

```

`rbp_eclip/test_subset.txt`: 
```
AARS
```

## Reproducible script

Regardless of which approch you choose to take, consider writing a single script/Makefile in the model-group root (at the same directory level as `template/`). The script/Makefile should generate or softlink all the files given the template folder, making it easier to update the files later.

<!-- - `generate.bash` -->
<!-- - `make all` -->
<!-- - `snakemake` -->
