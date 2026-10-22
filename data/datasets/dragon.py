from typing import TypeVar, Union, Literal, ClassVar, Generic, Any, Callable
from pydantic import BaseModel
import os
from abc import ABC, abstractmethod
from uuid import uuid4
from datasets import Dataset

from common.types import PathLike, Path
from data.sources import Source, SourceConfig, SourceKind, source_factory

SourceType = TypeVar('SourceType', covariant=True)
ItemDtype = TypeVar('BatcherType', covariant=True)


class _Dataset(Dataset, ABC, Generic[SourceType, ItemDtype]):
    kind: ClassVar[str]
    _registry: ClassVar[dict[str, type["_Dataset"]]] = {}

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if hasattr(cls, "kind"):
            _Dataset._registry[cls.kind] = cls
        else:
            _Dataset._registry[cls.__name__] = cls

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, identifier: Union[int, str]) -> ItemDtype: ...


class DragonDataset(Generic[SourceType, ItemDtype]):
    def __init__(self, source_kind: SourceKind, source_cfg: SourceConfig):
        self.source = source_factory(source_cfg, kind=source_kind)
        self.kind = source_kind
        self.config = source_cfg

    def __len__(self) -> int:
        import numpy as np
        return np.floor(len(self.source)/self.config.batcher_config.batch_size).item()

    def __getitem__(self, identifier: Union[str, int]) -> ItemDtype:
        return self.source.read().__getitem__(identifier)
