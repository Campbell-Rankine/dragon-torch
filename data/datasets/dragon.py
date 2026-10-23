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
    data: ClassVar[Any] = None
    _registry: ClassVar[dict[str, type["_Dataset"]]] = {}

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if hasattr(cls, "kind"):
            _Dataset._registry[cls.kind] = cls
        else:
            _Dataset._registry[cls.__name__] = cls

    def describe(self, title: str = "Dragon Dataset Base Class", logger=None, level='info') -> None:
        if logger is not None:
            recorder = getattr(logger, level)
        else:
            recorder = print

        recorder(f"Dataset Title={title}")
        for key, value in self.config.__dict__.items():
            recorder(f"    {key.capitalize()}={value}")

    def set_data(self, _d):
        setattr(self, 'data', _d)

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, identifier: Union[int, str]) -> ItemDtype: ...


class DragonDataset(_Dataset[SourceType, ItemDtype], Generic[SourceType, ItemDtype]):
    def __init__(self, source_kind: SourceKind, source_cfg: SourceConfig):
        self.source = source_factory(source_cfg, kind=source_kind)
        self.kind = source_kind
        self.config = source_cfg

    def post_process(self, x):
        return x if not hasattr(self.config, 'post_process') else self.config.post_process(x)

    def load(self):
        setattr(self, 'data', self.source.read())

    def get_dataset(self) -> ItemDtype:
        return self.data if not self.data is None else self.source.read()

    def __len__(self) -> int:
        import numpy as np
        return np.floor(len(self.source)/self.config.batcher_config.batch_size).item()

    def __getitem__(self, identifier: Union[str, int]) -> ItemDtype:
        return self.post_process(self.self.get_dataset().__getitem__(identifier))
