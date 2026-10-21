from typing import Union, Literal, Any, Callable, Annotated
from pydantic import BaseModel, AfterValidator
import torch as T
from functools import cache
import os
import numpy as np
from data.sources.base import Source, SourceConfig, SourceOutput
from common.types import DatasetSplitType, DeviceLike, TorchNumeric, PathLike, Path
from common.defaults import identity

type HF_Formats = Union[Literal['torch']]
type HFSourceKind = Union[Literal['hugging-face'],
                          Literal['hugging-face-polars']]


class HFSourceConfig(SourceConfig):
    name: str
    repository: str
    options: dict[str, Any]
    device: DeviceLike
    dtype: TorchNumeric
    split: DatasetSplitType
    download_path: PathLike
    post_process: Callable = identity
    _fmt: HF_Formats = 'torch'
    kind: HFSourceKind = 'hugging-face'
    bearer: str = None
    keep_dataset_in_memory: bool = False


class HFPolarsSourceConfig(HFSourceConfig):
    kind: HFSourceKind = 'hugging-face-polars'
    as_tensor: bool = False

### Output types + Validation ###


class _HFSourceOutput(SourceOutput[dict[str, T.Tensor]]):
    device: DeviceLike
    dtype: TorchNumeric


def apply_device_and_dtype(x: _HFSourceOutput) -> _HFSourceOutput:
    result = {}
    for k, v in x.items():
        result[k] = v.to(x.device)
        result[k] = result[k].to(dtype=getattr(T, x.dtype))
    x.update(result)
    return x


HFSourceOutput = Annotated[_HFSourceOutput,
                           AfterValidator(lambda x: apply_device_and_dtype(x))]


class _HFPolarsOutput(SourceOutput[dict[str, Union[T.Tensor, np.ndarray]]]):
    as_tensor: bool = False
    device: DeviceLike = None
    dtype: TorchNumeric = None
    kind: str = 'hugging-face-polars'


def convert_np_to_tensor(x: _HFPolarsOutput) -> _HFPolarsOutput:
    if not x.as_tensor:
        return x
    result = {}
    for k, v in x.items():
        result[k] = T.from_numpy(v).to(x.device, dtype=getattr(T, x.dtype))
    x.update(result)
    return x


HFPolarsOutput = Annotated[_HFPolarsOutput,
                           AfterValidator(lambda x: convert_np_to_tensor(x))]

### HF Source Classes ###


@cache
def load_hf_dataset(cfg: Union[HFSourceConfig, HFPolarsSourceConfig]):
    from datasets import load_dataset
    return load_dataset(cfg.repository, split=cfg.split, cache_dir=cfg.download_path, keep_in_memory=cfg.keep_dataset_in_memory, **cfg.options)


class HuggingFaceSource(Source[HFSourceOutput]):
    kind: str = "hugging-face"

    def __init__(self, cfg: HFSourceConfig):
        self.cfg = cfg

    def _query_hf_dataset_length(self) -> int:
        import requests
        url = f"https://datasets-server.huggingface.co/size?dataset={self.cfg.name}"
        response = requests.get(url).json() if self.cfg.bearer is None else requests.get(
            url, headers={'Authorization': self.cfg.bearer}).json()
        return response['size']['splits'][self.cfg.split]['num_rows']

    def __len__(self):
        return self._query_hf_dataset_length()

    def __getitem__(self, identifier: Any) -> HFSourceOutput:
        ds = load_hf_dataset(self.cfg).__getitem__(identifier)
        output = HFSourceOutput(
            **ds, identifier=identifier, device=self.cfg.device, dtype=self.cfg.dtype)
        return output


class HuggingFacePolarsSource(Source[HFPolarsOutput]):
    kind: str = "hugging-face-polars"

    def __init__(self, cfg: HFPolarsSourceConfig):
        self.cfg = cfg

    def _query_hf_dataset_length(self) -> int:
        import requests
        url = f"https://datasets-server.huggingface.co/size?dataset={self.cfg.name}"
        response = requests.get(url).json() if self.cfg.bearer is None else requests.get(
            url, headers={'Authorization': self.cfg.bearer}).json()
        return response['size']['splits'][self.cfg.split]['num_rows']

    def __len__(self):
        return self._query_hf_dataset_length()

    def __getitem__(self, identifier: Any) -> HFPolarsOutput:
        ds = load_hf_dataset(self.cfg).__getitem__(identifier)
        output = HFPolarsOutput(**ds, identifier=identifier, as_tensor=self.cfg.as_tensor,
                                device=self.cfg.device, dtype=self.cfg.dtype)
        return output
