from typing import TypeVar, Union, Literal, ClassVar, Generic, Any, Callable
from pydantic import BaseModel
import os
from common.types import PathLike, Path
from abc import ABC, abstractmethod
from uuid import uuid4
from templates.base_classes import DragonBaseClass

SourceType = TypeVar('SourceType', covariant=True)
OutputType = TypeVar('OutputType', covariant=True)

type ItemType = Union[str, int]
type BatchType = list[Union[str, int]]
# Base source class


def source_factory(cfg, kind=None):
    if not hasattr(cfg, 'kind') and kind is None:
        raise ValueError(
            f"Unable to figure out source kind from cfg={cfg}, cls=None")
    _kind = cfg.kind if hasattr(cfg, 'kind') else kind
    return Source._registry[_kind](cfg)


class Source(DragonBaseClass, Generic[OutputType]):

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


class DiskSource(Source[SourceOutput[SourceType]], Generic[SourceType]):
    kind: str = "disk"

    def __init__(self, config: SourceConfig):
        self.config = config
        self.file_length = None

    def _load_from_dir(self) -> dict:
        result = {}
        for fpath in os.listdir(self.config.loc):
            name: str = fpath.split('/')[-1].split('.')[0]
            with open(fpath, 'rb' if self.config.read_bytes else 'r') as file:
                result[name] = self.config.load_fn(file)
        return result

    def _load_from_single_file(self) -> dict:
        _fpath = self.config.loc if isinstance(
            self.config.loc, Path) else Path(self.config.loc)
        with open(_fpath, 'rb' if self.config.read_bytes else 'r') as file:
            return self.config.load_fn(file)

    def _get_length_for_directory(self) -> int:
        return len(os.listdir(self.config.loc))

    def _get_length_for_single_file(self) -> int:
        if not self.file_length is None:
            return self.file_length
        else:
            _fpath = self.config.loc if isinstance(
                self.config.loc, Path) else Path(self.config.loc)
            with open(_fpath, 'rb' if self.config.read_bytes else 'r') as file:
                self.file_length = len(self.config.load_fn(file))
                return self.file_length

    def __len__(self) -> int:
        match self.config.single_file:
            case True:
                return self._get_length_for_single_file()
            case False:
                return self._get_length_for_directory()

    def read(self):
        data = self._load_from_dir() if not self.config.single_file else self._load_from_single_file()
        return data
