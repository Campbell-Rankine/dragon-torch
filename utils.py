import numpy as np
from uuid import uuid4
from common.types import Path, PathLike
from typing import Union, Literal, Iterable


class IndexCounter(Iterable):
    _val = -1

    def __iter__(self):
        self._val = self._val + 1
        yield self._val

    def __next__(self):
        return iter(self)

    def __call__(self):
        return int(next(iter(self)))


def initialize_dataset_type(type_: Union[dict, list]):
    if type_ == dict:
        return {}, lambda x: str(uuid4())
    elif type_ == list:
        return [], IndexCounter()


def gen_random_file_source(fpath: PathLike, num_batches: int, shape: tuple[int, int], type_: type = dict) -> Path:
    if not isinstance(fpath, Path):
        fpath = Path(fpath)
    dataset, index_fn = initialize_dataset_type(type_=type_)
    for i in range(num_batches):
        rng = np.random.default_rng(seed=42)

        # Generate a 100x4 dataset (100 samples, 4 features)
        # Mean = 0, Standard Deviation = 1
        matrix_dataset = rng.normal(loc=0.0, scale=1.0, size=shape)
        dataset[index_fn()] = matrix_dataset
    fpath.touch(exist_ok=True)
    np.save(fpath, dataset)
    return fpath


def gen_random_directory_source(source_dir, num_batches: int, shape: tuple[int, int], ext_: str = '.npz', type_: type = dict) -> Path:
    source_path: Path = source_dir if isinstance(
        source_dir, Path) else Path(source_dir)
    source_path.mkdir(exist_ok=True, parents=True)

    _, index_fn = initialize_dataset_type(type_=type_)
    for i in range(num_batches):
        rng = np.random.default_rng(seed=42)

        # Generate a 100x4 dataset (100 samples, 4 features)
        # Mean = 0, Standard Deviation = 1
        matrix_dataset = rng.normal(loc=0.0, scale=1.0, size=shape)
        batch_path: Path = source_path / index_fn()
        batch_path.touch()

        np.save(batch_path, matrix_dataset)
    return source_path
