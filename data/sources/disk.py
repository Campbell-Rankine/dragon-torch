
from functools import partial
from common.types import PathLike, Path
from typing import ClassVar
from .base import Source, SourceConfig, SourceKind, source_factory, OutputType


class FileSourceConfig(SourceConfig):
    single_file: bool = True
    kind: str = 'file'


class FileSource(Source):
    kind: str = 'file'

    def __init__(self, cfg: FileSourceConfig):
        self.cfg = cfg
        self.__length: int = None
        self.__shape: tuple[int, int] = None

    def set_read_option(self, key, value):
        setattr(self, 'read', partial(self.read, **{key: value}))

    def read(self, **kw) -> OutputType:
        return self.cfg.dtype(self.cfg.load_fn(self.cfg.loc, **kw))

    def shape(self) -> tuple[int]:
        if self.__shape is not None:
            return self.__shape
        __data = self.read()
        self.__length = len(__data) if hasattr(
            __data, '__len__') else __data.shape[0]
        self.__shape = getattr(__data, 'shape', None)
        del __data
        return self.__shape

    def __len__(self) -> int:
        if self.__length is not None:
            return self.__length
        __data = self.read()
        self.__length = len(__data) if hasattr(
            __data, '__len__') else __data.shape[0]
        self.__shape = getattr(__data, 'shape', None)
        del __data
        return self.__length


class DirectorySourceConfig(SourceConfig):
    single_file: bool = False
    kind: str = 'directory'
    batch_size: int = None


class DirectorySource(Source):
    kind: ClassVar[str] = 'directory'

    def __init__(self, cfg: DirectorySourceConfig, batched: bool = False):
        self.cfg = cfg
        self.batched = batched

    def read(self) -> OutputType:
        return self.cfg.dtype(self.cfg.load_fn(self.cfg.loc))

    def __len__(self) -> int:
        import os
        return len(os.listdir(self.cfg.loc)) if not self.batched else len(os.listdir(self.cfg.loc))*self.cfg.batch_size
