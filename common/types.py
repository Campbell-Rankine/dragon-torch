import os
from dotenv import load_dotenv
from typing import Union, Literal, List
from .defaults import numpy_default_dtype
load_dotenv('./.env')

# ------- DatasetDefinition ------- #
type BackendTypes = Union[Literal['jax'],
                          Literal['numpy'], Literal['pandas'], Literal['torch']]

# ------- NumpyDataset Types ------- #

DefaultNumpyDtype = numpy_default_dtype
type NumpyDatasetAllowedDtypes = Union[Literal['float64'],
                                       Literal['float32'], Literal['float16'], Literal['int32'], Literal['int64']]


# ------- Torch Types ------- #
type DeviceLike = Union[Literal['cpu'], Literal['cuda']]
type FileExtensionLike = List[str | None]
