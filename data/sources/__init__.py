from .base import Source, SourceOutput, SourceConfig, DiskSource, source_factory
from .hf import HFSourceConfig, HuggingFaceSource, HFSourceOutput
from typing import Literal, Union

type SourceKind = Union[Literal['disk'], Literal['hugging-face']]
