from data.datasets.dragon import DragonDataset, SourceConfig, SourceType, SourceKind, ItemDtype
import torch
import os
from dotenv import load_dotenv
from common.types import TorchNumeric, DeviceLike, DatasetSplitType, PathLike
from data.sources.hf import HuggingFaceSource, HFSourceConfig
from typing import Union, Literal, Any

load_dotenv('./.env')

type SemasiaItemDtype = dict[str, torch.Tensor]


class SemasiaSourceConfig(HFSourceConfig):
    name: str = "semasia-celeba"
    repository: str = "spaicom-lab/semasia-celeba"
    options: list = []
    device: DeviceLike = 'cpu'
    dtype: TorchNumeric = 'float64'
    split: DatasetSplitType = 'train'
    download_path: PathLike = os.getenv('DATASET_CACHE_DIR')

    def set_model(self, model_id: str):
        setattr(self, 'options', [model_id])


class SemasiaCeleba(DragonDataset[HuggingFaceSource, SemasiaItemDtype]):
    def __init__(self, cfg: SemasiaSourceConfig):
        super().__init__(source_kind='hugging-face', source_cfg=cfg)

    def set_model(self, model_id: str):
        if len(self.config.options) >= 1:
            raise AttributeError(
                f"Unable to set model name as it was previously set to: {self.config.options}")
        else:
            self.config.set_model(model_id)

    def __getitem__(self, identifier: Union[str, int]) -> SemasiaItemDtype:
        return super().__getitem__(identifier)


class SemasiaCelebaSingleModel(SemasiaCeleba):
    def __init__(self, model_name: str, cfg: SemasiaSourceConfig):
        super().__init__(cfg)
        self.set_model(model_name)
