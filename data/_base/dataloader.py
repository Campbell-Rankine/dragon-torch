from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import LoaderConfig
from .dataset import BaseDataset


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
