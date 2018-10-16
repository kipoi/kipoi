# model.py
Custom models enable using any other framework or non-deep learning predictive model to be integrated within Kipoi. In general it is highly advisable not to use custom models if there is an implementation for the model that should be integrated, in other words: If your model is a pytorch model, please use the pytorch model type in Kipoi rather than defining your own custom model type.

Also, custom models should never deviate from using only numpy arrays, lists thereof, or dictionaries thereof as input for the `predict_on_batch` function. This is essential to maintain a homogeneous and clear interface between dataloaders and models in the Kipoi zoo!


The use of a custom model requires definition of a Kipoi-compliant model object, which can then be referred to by the model.yaml file. The model class has to be a subclass of `BaseModel` defined in `kipoi.model`, which in other words means that `def predict_on_batch(self, x)` has to be implemented. So for example if `batch` is  what the dataloader returns for a batch then `predict_on_batch(batch['inputs'])` has to run the model prediction on the given input.

A very simple version of such a model definition that can be stored in for example `model.py` may be:

```python
from kipoi.model import BaseModel

class MyModel(BaseModel):
    def __init__(self, file_path):
        ...
        self.model = load_model_parameters(file_path)

    # Execute model prediction for input data
    def predict_on_batch(self, x):
        return self.model.predict(x)
```

This can then be integrated in the model.yaml in the following way:

```yaml
defined_as: model.MyModel
args:
  file_path: # get model parameters from an url
    url: https://zenodo.org/path/to/my/architecture/file
    md5: ....
...
```
