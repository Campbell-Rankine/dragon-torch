import os
from utils import gen_random_directory_source, gen_random_file_source, np, Path, PathLike


class TestBlobFileHandler:
    def __init__(self, blob_shape: tuple[int, int], num_batches: int = 10):
        self.blob_shape = blob_shape
        self.num_batches = num_batches

    def enter(self, fpath: Path) -> Path:

        return gen_random_file_source(
            fpath, num_batches=self.num_batches, shape=self.blob_shape)

    def exit(self, fpath: Path):
        os.removedirs(fpath)

    def __call__(self, test_fn, dataset_path: PathLike, **test_kw):
        dir_or_file_path = self.enter(dataset_path)
        result = test_fn(dataset_path, **test_kw)
        self.exit(dir_or_file_path)
        return result
