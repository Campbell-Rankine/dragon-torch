from pydantic import BaseModel, AfterValidator
import os
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Any, Callable, ClassVar, Generic, Literal, Optional, TypeVar, Union

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from torch.utils.data import DataLoader, Dataset
from common.utils import build_path, PathLike

# --------------------------------------------------------------------------- #
# Configs (pydantic v2): frozen + extra="forbid" so typos fail loudly
# --------------------------------------------------------------------------- #


class _Config(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_default=True)


type DiskSourceKind = Literal['disk']


class _DiskSourceConfig(_Config):
    kind: DiskSourceKind = "disk"
    root: PathLike
    pattern: str = "**/*"
    extensions: Optional[tuple[str, ...]] = None

    @field_validator("root")
    @classmethod
    def _root_exists(cls, v: Path) -> Path:
        v = v.expanduser().resolve()
        if not v.is_dir():
            raise ValueError(f"root directory does not exist: {v}")
        return v


type HTTPSourceKind = Literal['http']


class HTTPSourceConfig(_Config):
    kind: HTTPSourceKind = "http"
    base_url: str
    # relative paths appended to base_url
    keys: tuple[str, ...] = Field(min_length=1)
    timeout_s: float = Field(30.0, gt=0)
    max_retries: int = Field(3, ge=0)
    backoff_factor: float = Field(0.5, ge=0)
    auth_token: Optional[SecretStr] = None  # never leaks in repr/logs

    @field_validator("base_url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v.rstrip("/")


# Discriminated union: pydantic picks the right model from `kind`, with clear errors.
DiskSourceConfig = Annotated[_DiskSourceConfig,
                             AfterValidator(lambda x: build_path(x.root))]
SourceConfig = Annotated[Union[DiskSourceConfig,
                               HTTPSourceConfig], Field(discriminator="kind")]


class LoaderConfig(_Config):
    batch_size: int = Field(32, ge=1)
    shuffle: bool = False
    num_workers: int = Field(0, ge=0)
    pin_memory: bool = False
    drop_last: bool = False
    prefetch_factor: Optional[int] = Field(None, ge=1)
    persistent_workers: bool = False
    seed: int = 0

    @model_validator(mode="after")
    def _worker_consistency(self) -> "LoaderConfig":
        # torch raises on these combos with num_workers == 0; catch them at config time
        if self.num_workers == 0 and (self.prefetch_factor is not None or self.persistent_workers):
            raise ValueError(
                "prefetch_factor / persistent_workers require num_workers > 0")
        return self


class DatasetConfig(_Config):
    """Subclass this for task-specific fields (image size, tokenizer, etc.)."""
    source: SourceConfig
    loader: LoaderConfig = LoaderConfig()
