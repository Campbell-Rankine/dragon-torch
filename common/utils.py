from pathlib import Path
from .types import PathLike


def build_path(x: PathLike) -> Path:
    return x if isinstance(x, Path) else Path(x)
