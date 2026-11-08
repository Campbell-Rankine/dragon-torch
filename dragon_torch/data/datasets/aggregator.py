from typing import Any, Callable, Iterator, Sequence, Union
import numpy as np
from .dragon import DragonDataset
from copy import deepcopy
from torch.utils.data import IterableDataset, get_worker_info

type WeightKey = Union[int, str]
type AggregatorBatch = list[list[Any]]

STREAMING_ATTRIBUTES = ['stream_from_hf']


class Aggregator(IterableDataset):
    """Combines map-style datasets into weighted batches.

    Every batch holds int(weight * batch_size) samples from each dataset, drawn
    without replacement, so an epoch ends as soon as any weighted dataset runs
    out. Datasets are indexed per sample: pass DragonDatasets without a batch_size.

    Weights are keyed by position in `datasets` or by a dataset's `kind`.
    Call set_epoch() between epochs to reshuffle.
    """

    def __init__(self, datasets: Sequence[DragonDataset], batch_size: int, weights: dict[WeightKey, float],
                 shuffle: bool = True, seed: int = None, collate_fn: Callable[[AggregatorBatch], Any] = None):
        if len(datasets) == 0:
            raise ValueError("Aggregator needs at least one dataset")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.datasets = self.validate_datasets(list(datasets))
        self.batch_size = batch_size
        self.weights = self._resolve_weights(weights)
        self.samples_per_dataset = [int(w * batch_size) for w in self.weights]
        if sum(self.samples_per_dataset) == 0:
            raise ValueError(
                f"weights={self.weights} draw no samples at batch_size={batch_size}")
        self.shuffle = shuffle
        self.seed = seed
        self.collate_fn = collate_fn
        # len() can be a network call for hugging face sources, so only ask once
        self._lengths = [len(ds) for ds in self.datasets]
        self.set_epoch(0)

    def validate_datasets(self, datasets: list[DragonDataset]) -> list[DragonDataset]:
        if not issubclass(datasets[0].__class__, DragonDataset):
            return datasets
        result = []
        for dataset in datasets:
            if not hasattr(dataset, 'to_aggregator') and not hasattr(dataset, 'config'):
                continue
            if hasattr(dataset, 'to_aggregator'):
                dataset: DragonDataset = dataset.to_aggregator()
            _cfg = dataset.config
            if hasattr(_cfg, 'stream_from_hf'):
                setattr(deepcopy(_cfg), 'stream_from_hf', False)
                setattr(dataset, 'config', _cfg)
                print(
                    f'MSG :: dataset={dataset.kind}, Setting stream_from_hf=False for ')
                result.append(dataset)
            else:
                result.append(dataset)
        return result

    def _index_of(self, key: WeightKey) -> int:
        if isinstance(key, int):
            if not 0 <= key < len(self.datasets):
                raise KeyError(
                    f"Weight key {key} out of range for {len(self.datasets)} datasets")
            return key
        matches = [i for i, ds in enumerate(self.datasets)
                   if getattr(ds, 'kind', None) == key]
        if len(matches) != 1:
            raise KeyError(
                f"Weight key '{key}' matches {len(matches)} datasets by kind, expected 1")
        return matches[0]

    def _resolve_weights(self, weights: dict[WeightKey, float]) -> list[float]:
        if len(weights) != len(self.datasets):
            raise ValueError(
                f"Expected {len(self.datasets)} weights, got {len(weights)}")
        resolved = [None] * len(self.datasets)
        for key, weight in weights.items():
            i = self._index_of(key)
            if resolved[i] is not None:
                raise ValueError(f"Dataset {i} was given more than one weight")
            if not 0 <= weight <= 1:
                raise ValueError(
                    f"Weight for dataset {i} must be in [0, 1], got {weight}")
            resolved[i] = float(weight)
        if sum(resolved) > 1 + 1e-9:
            raise ValueError(f"Weights sum to {sum(resolved)}, must be <= 1")
        return resolved

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        rng = np.random.default_rng(
            None if self.seed is None else [self.seed, epoch])
        self._orders = [rng.permutation(n) if self.shuffle else np.arange(n)
                        for n in self._lengths]

    def __len__(self) -> int:
        return min(n // k for n, k in zip(self._lengths, self.samples_per_dataset) if k > 0)

    def __getitem__(self, identifier: int) -> AggregatorBatch:
        n_batches = len(self)
        idx = identifier + n_batches if identifier < 0 else identifier
        if not 0 <= idx < n_batches:
            raise IndexError(
                f"Batch {identifier} out of range for {n_batches} batches")
        batch = [[ds[int(j)] for j in order[idx * k:(idx + 1) * k]]
                 for ds, order, k in zip(self.datasets, self._orders, self.samples_per_dataset)]
        return batch if self.collate_fn is None else self.collate_fn(batch)

    def __iter__(self) -> Iterator[AggregatorBatch]:
        # under a multi-worker DataLoader each worker takes every num_workers-th batch
        worker = get_worker_info()
        indices = range(len(self)) if worker is None else range(
            worker.id, len(self), worker.num_workers)
        for idx in indices:
            yield self[idx]
