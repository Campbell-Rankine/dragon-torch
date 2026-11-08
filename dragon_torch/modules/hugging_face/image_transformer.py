from dragon_torch.modules.base import DragonModule, ModelDtype
import torch
from typing import Generic


class HFImageTransformer(DragonModule[ModelDtype], Generic[ModelDtype]):
    def __init__(self, model_name: str, model_options: dict = {}):
        from transformers import AutoModel
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, **model_options)

    def forward(self, X: torch.Tensor, **kwargs) -> ModelDtype:
        return self.model(**X, **kwargs)


class HFImageTransformerPipeline(HFImageTransformer[ModelDtype], Generic[ModelDtype]):
    def __init__(self, model_name: str, model_options: dict, processor_name: str, processor_options: dict):
        from transformers import AutoImageProcessor
        super().__init__(model_name=model_name, model_options=model_options)
        self.processor = AutoImageProcessor.from_pretrained(
            processor_name, **processor_options)

    def forward(self, X: torch.Tensor, **kwargs) -> ModelDtype:
        inputs = self.processor(images=X, return_tensors='pt')
        return super().forward(inputs, **kwargs)
