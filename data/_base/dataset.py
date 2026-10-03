from pydantic import BaseModel
from typing import Union, Literal, Annotated
import os

"""Base data-loading primitives: pydantic configs, pluggable sources, base dataset.
 
Suggested layout once this grows:
    mylib/data/config.py    -> configs
    mylib/data/sources.py   -> Source ABC, registry, DiskSource, HTTPSource
    mylib/data/datasets.py  -> BaseDataset
    mylib/data/loaders.py   -> build_dataloader
"""
from __future__ import annotations
import numpy as np
import torch
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Optional, TypeVar
from torch.utils.data import DataLoader, Dataset

from .sources import build_source
from .config import SourceConfig, LoaderConfig, _Config


# --------------------------------------------------------------------------- #
# Base dataset: subclasses only implement decode()
# --------------------------------------------------------------------------- #
ConfigT = TypeVar("ConfigT", bound=DatasetConfig)
SampleT = TypeVar("SampleT", covariant=True)


class DatasetConfig(_Config):
    """Subclass this for task-specific fields (image size, tokenizer, etc.)."""
    source: SourceConfig
    loader: LoaderConfig = LoaderConfig()


class BaseDataset(Dataset[SampleT], ABC, Generic[ConfigT, SampleT]):
    def __init__(self, config: ConfigT, transform: Optional[Callable[[Any], Any]] = None):
        self.config = config
        self.transform = transform
        self.source = build_source(config.source)
        self.keys = self.source.list_keys()
        if not self.keys:
            raise RuntimeError(
                f"No samples found for source config: {config.source!r}")

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> SampleT:
        key = self.keys[idx]
        sample = self.decode(self.source.read(key), key)
        return self.transform(sample) if self.transform else sample

    @abstractmethod
    def decode(self, raw: bytes, key: str) -> SampleT:
        """Turn raw bytes into a sample (tensor / dict of tensors / numpy)."""
        raise NotImplementedError(
            f"Not implemented for type={self.__class__.__name__}")

    def collate_fn(self, batch: list[SampleT]) -> Any:
        """Override for custom batching (padding, ragged data, jax arrays, ...)."""
        return torch.utils.data.default_collate(batch)


# --------------------------------------------------------------------------- #
# Loader factory: reproducible seeding, config -> DataLoader
# --------------------------------------------------------------------------- #
def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_dataloader(dataset: BaseDataset, cfg: Optional[LoaderConfig] = None) -> DataLoader:
    cfg = cfg or dataset.config.loader
    g = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        collate_fn=dataset.collate_fn,
        worker_init_fn=_seed_worker,
        generator=g,
    )
