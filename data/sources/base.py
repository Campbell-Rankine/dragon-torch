from typing import TypeVar, Union, Literal, ClassVar, Generic, Any, Callable
from pydantic import BaseModel
import os
from common.types import PathLike, Path
from abc import ABC, abstractmethod
from uuid import uuid4

SourceType = TypeVar('SourceType', covariant=True)
OutputType = TypeVar('OutputType', covariant=True)

type ItemType = Union[str, int]
type BatchType = list[Union[str, int]]
type SourceKind = Union[Literal['base'], Literal['directory'], Literal['file'],
                        Literal['hugging-face'], Literal['hugging-face-polars']]
# Base source class


def source_factory(cfg, kind=None):
    if not hasattr(cfg, 'kind') and kind is None:
        raise ValueError(
            f"Unable to figure out source kind from cfg={cfg}, cls=None")
    _kind = cfg.kind if hasattr(cfg, 'kind') else kind
    return Source._registry[_kind](cfg)


class Source(ABC, Generic[OutputType]):
    kind: ClassVar[str]
    _registry: ClassVar[dict[str, type["Source"]]] = {}

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if hasattr(cls, "kind"):
            Source._registry[cls.kind] = cls
        else:
            Source._registry[cls.__name__] = cls

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def read(self) -> OutputType: ...


# Standard disk source
class SourceConfig(BaseModel):
    loc: PathLike
    load_fn: Callable
    ext: str
    dtype: type = dict[str, Any]
    single_file: bool = False
    read_bytes: bool = False


class SourceOutput(dict[str, SourceType]):
    identifier: Union[str, int]
    _id: str = str(uuid4())
