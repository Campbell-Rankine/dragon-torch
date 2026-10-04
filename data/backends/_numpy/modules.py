from data._base.sources import Source
from .configs import NumpySourceConfig, NumpyLoaderConfig


class NumpyDiskSource(Source[NumpySourceConfig]):
    kind = "numpy-disk"
