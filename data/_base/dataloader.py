from pydantic import BaseModel
from typing import Union, Literal, Annotated
import os

"""Base data-loading primitives: pydantic configs, pluggable sources, base dataset.
 
Suggested layout once this grows:
    mylib/data/config.py    -> configs
    mylib/data/sources.py   -> Source ABC, registry, DiskSource, HTTPSource
    mylib/data/datasets.py  -> BaseDataset
    mylib/data/loaders.py   -> build_dataloader
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Any, Callable, ClassVar, Generic, Literal, Optional, TypeVar, Union

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from torch.utils.data import DataLoader, Dataset
from .config import LoaderConfig
