from common.torch.types import DeviceLike, TorchFloatKind, TorchIntegerKind
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Union, Literal, List
from .defaults import numpy_default_dtype
load_dotenv('./.env')


# ------- Global ------- #
type PathLike = Union[Path, str]

# ------- DatasetDefinition ------- #
type BackendTypes = Union[Literal['jax'],
                          Literal['numpy'], Literal['pandas'], Literal['torch']]
type DatasetSplitType = Union[Literal['train'],
                              Literal['test'], Literal['validation']]


# ------- Torch Types ------- #
type TorchNumeric = Union[TorchIntegerKind, TorchFloatKind]
type FileExtensionLike = List[str | None]
