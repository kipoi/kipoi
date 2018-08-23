## Plugins

To enable additional functionality beyond just running model predictions, there are two additional plugins available:

- kipoi-veff: variant effect prediction [github](https://github.com/kipoi/kipoi-veff), [docs](https://kipoi.org/veff-docs/)
- kipoi-interpret: model interpretation.[github](https://github.com/kipoi/kipoi-interpret), [docs](https://kipoi.org/interpret-docs/)

### [Kipoi-veff](https://kipoi.org/veff-docs/)

Kipoi-veff is a plugin specific to genomics. Models trained to predict various molecular phenotypes from DNA sequence can be used to assess the impact of genetic mutations or variants. The veff plugin allows you to take the VCF file — canonical file format for storing genetic variants — and obtain changes in model predictions due to the genetic variants/mutations changing the DNA sequence.

![veff](https://cdn-images-1.medium.com/max/1200/1*cm8Cq5uWnCXC_GNhUrQNKg.png)

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

