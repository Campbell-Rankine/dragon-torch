# dragon-torch

[![Tests](https://github.com/Campbell-Rankine/dragon-torch/actions/workflows/tests.yml/badge.svg?branch=dev)](https://github.com/Campbell-Rankine/dragon-torch/actions/workflows/tests.yml)

PyTorch Deep Learning Dev Modules

### Sources:

Define source for data in a dataset. Source.__getitem__() is cached and returns the full dataset. Supported types:

- disk
- hugging-face
- hugging-face-polars

### Datasets:

take a source as input. Dataset.__getitem__() returns an iterable of batches from the input source

**NOTE**: Dataset aggregators define batching rules / filters for combining two datasets

### Tests:

Pushes to `dev` run the full test suite in GitHub Actions; when every test passes, `dev` is merged into `main` automatically. Run them locally with:

```bash
uv run --extra test --extra torch-cpu --extra torch-ext pytest            # everything
uv run --extra test --extra torch-cpu --extra torch-ext pytest -m "not network"  # skip Hugging Face Hub tests
```

### TODO:

- [ ] : Setup Lazy Loading for hugging face sources.
