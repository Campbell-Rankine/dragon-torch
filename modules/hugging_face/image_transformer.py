from modules.base import DragonModule, ModelDtype
import torch
from typing import Generic


class HFImageTransformer(DragonModule[ModelDtype], Generic[ModelDtype]):
    def __init__(self, model_name: str, model_options: dict = {}):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name, **model_options)

    def _validate_dtype(self, X):
        if not isinstance(X, ModelDtype) and not issubclass(X, ModelDtype):
            X.to(dtype=ModelDtype)

    def forward(self, X: torch.Tensor, **kwargs) -> ModelDtype:
        return self._validate_dtype(self.model(**X, **kwargs))


class HFImageTransformerPipeline(HFImageTransformer[ModelDtype], Generic[ModelDtype]):
    def __init__(self, model_name: str, model_options: dict, processor_name: str, processor_options: dict):
        from transformers import AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained(
            processor_name, **processor_options)
        super().__init__(model_name=model_name, model_options=model_options)

    def forward(self, X: torch.Tensor, **kwargs) -> ModelDtype:
        inputs = self.processor(images=X, return_tensors='pt')
        return super().forward(inputs, **kwargs)
