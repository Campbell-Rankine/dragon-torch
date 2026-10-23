from data.datasets.dragon import DragonDataset
import torch
import os
from datasets import Dataset
from dotenv import load_dotenv

from data.sources.hf import HuggingFaceSource, HFSourceConfig
from data.datasets.hugging_face.semasia.config import SemasiaSourceConfig, extract_semasia_dataset
from typing import Union, Literal, Any

load_dotenv('./.env')

type SemasiaItemDtype = dict[str, torch.Tensor]


class SemasiaCelebaSourceConfig(SemasiaSourceConfig):
    name: str = "semasia-celeba"
    repository: str = "spaicom-lab/semasia-celeba"


class SemasiaCeleba(DragonDataset[HuggingFaceSource, SemasiaItemDtype]):
    kind = 'hf-semasia-cleba'

    def __init__(self, cfg: SemasiaSourceConfig):
        super().__init__(source_kind=self.kind, source_cfg=cfg)

    def set_model(self, model_id: str):
        if len(self.config.options) >= 1:
            raise AttributeError(
                f"Unable to set model name as it was previously set to: {self.config.options}")
        else:
            self.config.set_model(model_id)

    def __getitem__(self, identifier: Union[str, int]) -> SemasiaItemDtype:
        return super().__getitem__(identifier)


class SemasiaCelebaSingleModel(SemasiaCeleba):
    kind = 'hf-semasia-celeba-single-model'

    def __init__(self, cfg: SemasiaSourceConfig, model_name: str = "aimv2_1b_patch14_224.apple_pt"):
        super().__init__(cfg)
        self.model_name = model_name
        self.set_model(model_name)
