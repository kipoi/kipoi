## Plugins

To enable additional functionality beyond just running model predictions, a plugin is available:

- kipoi-interpret: model interpretation.[github](https://github.com/kipoi/kipoi-interpret), [docs](https://kipoi.org/interpret-docs/)

### [Kipoi-interpret](https://kipoi.org/interpret-docs/)

Kipoi-interpret is a general (genomics agnostic) plugin and allows to compute the feature importance scores like saliency maps or DeepLift for Kipoi models.

```python
import kipoi
from kipoi_interpret.importance_scores.gradient import GradientXInput

model = kipoi.get_model("DeepBind/Homo_sapiens/TF/D00765.001_ChIP-seq_GATA1")

val = GradientXInput(model).score(seq_array)[0]
seqlogo_heatmap(val, val.T)
```

![interpret](https://cdn-images-1.medium.com/max/800/0*VWI94BzEZXRWc1Oe)

