import torch
import numpy as np
from pydantic import BaseModel
from typing import Union, Literal, Annotated
import os
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from common.types import DeviceLike, NumpyDatasetAllowedDtypes, DefaultNumpyDtype
from data._base.dataset import BaseDataset, SampleT, ConfigT
from data._base.config import DatasetConfig
from data.backends._numpy.types import NumpyOnlyOutputDtypes, NumpyTorchOutputDtypes

# output types
type NumpyDatasetOutputDtypes = Union[NumpyOnlyOutputDtypes,
                                      NumpyTorchOutputDtypes]

# config types
type NumpyFloatingDatasetConfigType = Union[NumpyFloat64DatasetConfig,
                                            NumpyFloat32DatasetConfig, NumpyFloat16DatasetConfig]
type NumpyIntegerDatasetConfigType = Union[NumpyInt32DatasetConfig,
                                           NumpyInt64DatasetConfig]
type NumpyDatasetConfigType = Union[NumpyDatasetConfig,
                                    NumpyFloatingDatasetConfigType, NumpyIntegerDatasetConfigType]


# factory
def build_numpy_dataset(cfg: NumpyDatasetConfigType) -> NumpyDataset:
    return NumpyDataset._registry[cfg.output_type.__name__+'-'+cfg.dtype](cfg)

### - NUMPY ONLY DATASETS - ###


class NumpyDatasetConfig(DatasetConfig):
    output_type: NumpyDatasetOutputDtypes
    dtype: NumpyDatasetAllowedDtypes = DefaultNumpyDtype


class _NumpyDataset(BaseDataset[NumpyDatasetConfig, np.ndarray]):
    dtype: ClassVar[str]
    output_type: ClassVar[type]
    _registry: ClassVar[dict[str, type["_NumpyDataset"]]] = {}

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if hasattr(cls, "output_type") and hasattr(cls, "dtype"):
            _NumpyDataset._registry[cls.output_type.__name__ +
                                    '-'+cls.dtype] = cls

    def _apply_dtype(self, x: NumpyDatasetOutputDtypes):
        if isinstance(x, np.ndarray):
            return x.view(dtype=getattr(np, self.config.dtype))
        else:
            return x.to(getattr(torch, self.config.dtype.capitalize()))

    @abstractmethod
    def decode(self, raw: bytes, key: str) -> np.ndarray: ...


class NumpyDataset(_NumpyDataset):
    dtype: NumpyDatasetAllowedDtypes
    output_type: NumpyDatasetOutputDtypes = np.ndarray

    def decode(self, raw: bytes, key: str) -> np.ndarray:
        import io
        result: np.ndarray = np.load(io.BytesIO(raw))
        return self._apply_dtype(result)


class NumpyFloat64DatasetConfig(NumpyDatasetConfig):
    dtype: NumpyDatasetAllowedDtypes = "float64"
    output_type: NumpyOnlyOutputDtypes = np.floating


class NumpyFloat64Dataset(_NumpyDataset):
    dtype: NumpyDatasetAllowedDtypes = "float64"
    output_type: NumpyOnlyOutputDtypes = np.floating


class NumpyFloat32DatasetConfig(NumpyDatasetConfig):
    dtype: NumpyDatasetAllowedDtypes = "float32"
    output_type: NumpyOnlyOutputDtypes = np.floating


class NumpyFloat32Dataset(_NumpyDataset):
    dtype: NumpyDatasetAllowedDtypes = "float32"
    output_type: NumpyOnlyOutputDtypes = np.floating


class NumpyFloat16DatasetConfig(NumpyDatasetConfig):
    dtype: NumpyDatasetAllowedDtypes = "float16"
    output_type: NumpyOnlyOutputDtypes = np.floating


class NumpyFloat16Dataset(_NumpyDataset):
    dtype: NumpyDatasetAllowedDtypes = "float16"
    output_type: NumpyOnlyOutputDtypes = np.floating


class NumpyInt32DatasetConfig(NumpyDatasetConfig):
    dtype: NumpyDatasetAllowedDtypes = 'int32'
    output_type: NumpyOnlyOutputDtypes = np.integer


class NumpyInt32Dataset(_NumpyDataset):
    dtype: NumpyDatasetAllowedDtypes = 'int32'
    output_type: NumpyOnlyOutputDtypes = np.integer


class NumpyInt64DatasetConfig(NumpyDatasetConfig):
    dtype: NumpyDatasetAllowedDtypes = 'int64'
    output_type: NumpyOnlyOutputDtypes = np.integer


class NumpyInt64Dataset(_NumpyDataset):
    dtype: NumpyDatasetAllowedDtypes = 'int64'
    output_type: NumpyOnlyOutputDtypes = np.integer


### - NUMPY -> TORCH DATASETS - ###
