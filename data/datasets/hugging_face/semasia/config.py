from data.sources.hf import HFSourceConfig
from common.types import TorchNumeric, DeviceLike, DatasetSplitType, PathLike
import os
import torch
from datasets import Dataset


def extract_semasia_dataset(ds: Dataset) -> dict[str, torch.Tensor]:
    return {'embeddings': torch.vstack(list(ds['embedding'])), 'labels': torch.Tensor(ds['celeb_id'])}


class SemasiaSourceConfig(HFSourceConfig):
    name: str
    repository: str
    options: list = []
    device: DeviceLike = 'cpu'
    dtype: TorchNumeric = 'float64'
    split: DatasetSplitType = 'train'
    download_path: PathLike = os.getenv('DATASET_CACHE_DIR')
    post_process = extract_semasia_dataset

    def set_model(self, model_id: str):
        setattr(self, 'options', [model_id])
