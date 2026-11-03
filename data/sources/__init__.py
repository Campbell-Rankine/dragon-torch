from .base import Source, SourceOutput, SourceConfig, source_factory
from .hf import HFSourceConfig, HuggingFaceSource, HFSourceOutput
from .disk import FileSourceConfig, FileSource, DirectorySourceConfig, DirectorySource
from typing import Literal, Union

type SourceKind = Union[Literal['file'], Literal['directory'],
                        Literal['hugging-face'], Literal['hugging-face-polars']]
