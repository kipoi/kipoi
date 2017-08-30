This docs shows how our API should ideally look like.

## CLI definition

Main arguments

- `<model>`: can be a git repository, local file path, remote file path or a short model string.

### Pre-process

Returns an hdf5 array.

```
model_zoo preproc <model> <preprocessor inputs...> -o <output.h5>
```

### Predict

Run the prediction.

```
model_zoo predict <model> <preprocessor inputs...> -o <output>
```

where `<model>` can be the directory containing the required files

### Score variants

```
model_zoo score_variants <model> <preprocessor inputs...> <vcf file> -o <output>
```

### Pull the model

Downloads the directory associated with the model

```
model_zoo pull <model> -o path
```

### Push the model

- Maybe use the tools from docker?

```
model_zoo push <model-dir>
```

### Test the model

Runs a set of unit-tests for the model

```
model_zoo test <model>
```
