from data._base.config import DatasetConfig
from typing import Union, NewType, Literal, Annotated

type AudioFileExtensionType = Union[Literal['wav'], Literal['mp3']]
type AudioSourceKind = Union[Literal['disk'], Literal['disk-multiple']]


class _AudioDatasetConfig(DatasetConfig):
    sample_rate: int
    file_ext: AudioFileExtensionType
    channels: int = 2
    source_kind: AudioSourceKind
