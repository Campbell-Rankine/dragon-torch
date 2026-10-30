import pandas as pd
from numpy import ndarray
from common.types import PathLike, Path
from .base import Source, SourceConfig, SourceKind, source_factory, OutputType


class FileSourceConfig(SourceConfig):
    single_file = True
    kind = 'file'


class FileSource(Source):
    kind: str = 'file'

    def __init__(self, cfg: FileSourceConfig):
        self.cfg = cfg
        self.__length: int = None

    def read(self) -> OutputType:
        return self.cfg.dtype(self.cfg.load_fn(self.cfg.loc))

    def __len__(self) -> int:
        if self.__length is not None:
            return self.__length
        __data = self.read()
        self.__length = len(__data) if hasattr(
            __data, '__len__') else __data.shape[0]
        del __data
        return self.__length


class DiskSourceConfig(SourceConfig):
    single_file = False
    kind = 'directory'
    batch_size = None


class DiskSource(Source):
    def __init__(self, cfg: DiskSourceConfig, batched: bool = False):
        self.cfg = cfg
        self.batched = batched

    def read(self) -> OutputType:
        return self.cfg.dtype(self.cfg.load_fn(self.cfg.loc))

    def __len__(self) -> int:
        import os
        return len(os.listdir(self.cfg.loc)) if not self.batched else len(os.listdir(self.cfg.loc))*self.cfg.batch_size
