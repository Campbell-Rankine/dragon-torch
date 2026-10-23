import os
from typing import Any, Callable
from models.base import DragonModel, DragonModelConfig, ModelDtype
from modules.hugging_face.image_transformer import HFImageTransformerPipeline


class HFDragonModelConfig(DragonModelConfig):
    hf_repo: str
    processor_repo: str = None
    model_options: dict[str, Any] = {}
    processor_options: dict[str, Any] = {}


class HFDragonImageEmbeddingModel(DragonModel[ModelDtype]):
    def __init__(self, loss_fn: Callable, cfg: HFDragonModelConfig):
        assert not cfg.processor_repo is None, f"Embedding models need initial pretrained processors"
        super().__init__(
            module=HFImageTransformerPipeline(
                model_name=cfg.hf_repo,
                model_options=cfg.model_options,
                processor_name=cfg.processor_repo,
                processor_options=cfg.processor_options
            ),
            loss_fn=loss_fn,
            cfg=cfg
        )
