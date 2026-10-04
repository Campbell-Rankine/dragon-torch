import numpy as np
from typing import Annotated, Callable, Literal
from common.types import PathLike
from data._base.config import DiskSourceConfig, LoaderConfig

type NumpyDiskSourceKind = Literal['numpy-disk']


class NumpySourceConfig(DiskSourceConfig):
    root: PathLike
    kind: NumpyDiskSourceKind = 'numpy-disk'
    extensions: tuple[str, ...] = ('npz', 'npy')


class NumpyLoaderConfig(LoaderConfig):
    load_fn: Callable = np.load
