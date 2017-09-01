This docs shows how our API should ideally look like.

## Load the library

```python
import modelzoo
```

## List all available models

Returns a pandas data.frame

```python
df = modelzoo.list_models()
df
```

Get all available information about some particular model (as python dictionary)

```python
modelzoo.model_info(df.iloc[1,"model"])
```

## Extracor

### Load the extractor

```python
Extractor = modelzoo.load_extractor("username/model1")
```
### Get information

```python
print(Extractor.__doc__)
```


### Initialize it

```python
preproc = Extractor(infile="~/file.txt", seq_len=10)
```

### Draw a few batches

```python
x = next(preproc)
print(x)
```

## Model

### Load the model

Refer by `username/model`

```python
# Most recent model
model = modelzoo.load_model("username/model1")

# Specific version
model = modelzoo.load_model("username/model1:v0.1")

# From directory
model = modelzoo.load_model("~/mymodels/model1")

# From github
model = modelzoo.load_model("https://github.com/kipoi/model-zoo/tree/master/examples/extended_coda")
```

### Run predictions

```python
y = model.predict(preproc)
```

## Extractor + Model bundle

```python
Me = modelzoo.pieline.ModelExtractor("username/model1")

## Get info
print(Me.__doc__)

## Setup
me = Me(infile="~/file.txt", seq_len=10)

y = me.predict()
```
