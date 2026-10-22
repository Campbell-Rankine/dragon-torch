# dragon-torch
PyTorch Deep Learning Dev Modules

### Sources:

Define source for data in a dataset. Source.__getitem__() is cached and returns the full dataset. Supported types:

- disk
- hugging-face
- hugging-face-polars

### Datasets:

take a source as input. Dataset.__getitem__() returns an iterable of batches from the input source

**NOTE**: Dataset aggregators define batching rules / filters for combining two datasets
