import os
import numpy as np
import pytest
from pydantic import ValidationError
from data.sources.base import Source, source_factory
from data.sources.disk import DirectorySource, DirectorySourceConfig
from utils import gen_random_directory_source

blob_shape = (4, 10, 2)  # BATCH_SIZE, XDim, YDim


def load_dir(path):
    # one entry per batch file, keyed by filename stem
    return {os.path.splitext(f)[0]: np.load(os.path.join(path, f))
            for f in sorted(os.listdir(path))}


class CountingLoader:
    """load_fn wrapper that records how often the source hits disk."""

    def __init__(self, fn=load_dir):
        self.fn = fn
        self.calls = 0

    def __call__(self, path):
        self.calls += 1
        return self.fn(path)


@pytest.fixture
def dir_dataset_path(tmp_path):
    return gen_random_directory_source(tmp_path / 'blobs', num_batches=blob_shape[0],
                                       shape=blob_shape[1:], ext_='.npy')


@pytest.fixture
def source(dir_dataset_path):
    return DirectorySource(DirectorySourceConfig(loc=dir_dataset_path, load_fn=load_dir))


class TestDirectorySourceConfig:
    def test_defaults(self, dir_dataset_path):
        cfg = DirectorySourceConfig(loc=dir_dataset_path, load_fn=load_dir)
        assert cfg.single_file == False
        assert cfg.kind == 'directory'
        assert cfg.batch_size is None
        assert cfg.read_bytes == False

    def test_batch_size(self, dir_dataset_path):
        cfg = DirectorySourceConfig(loc=dir_dataset_path, load_fn=load_dir,
                                    batch_size=8)
        assert cfg.batch_size == 8

    def test_accepts_str_loc(self, dir_dataset_path):
        cfg = DirectorySourceConfig(loc=str(dir_dataset_path), load_fn=load_dir)
        assert str(cfg.loc) == str(dir_dataset_path)

    def test_requires_loc(self):
        with pytest.raises(ValidationError):
            DirectorySourceConfig(load_fn=load_dir)

    def test_requires_callable_load_fn(self, dir_dataset_path):
        with pytest.raises(ValidationError):
            DirectorySourceConfig(loc=dir_dataset_path, load_fn='np.load')


class TestDirectorySourceInit:
    def test_is_source(self, source):
        assert isinstance(source, Source)

    def test_not_batched_by_default(self, source):
        assert source.batched == False

    def test_init_does_not_read(self, dir_dataset_path):
        loader = CountingLoader()
        DirectorySource(DirectorySourceConfig(loc=dir_dataset_path, load_fn=loader))
        assert loader.calls == 0

    def test_source_factory(self, dir_dataset_path):
        cfg = DirectorySourceConfig(loc=dir_dataset_path, load_fn=load_dir)
        assert isinstance(source_factory(cfg), DirectorySource)


class TestDirectorySourceRead:
    def test_load_fn_receives_directory(self, dir_dataset_path):
        seen = []
        cfg = DirectorySourceConfig(loc=dir_dataset_path,
                                    load_fn=lambda p: seen.append(p) or {})
        DirectorySource(cfg).read()
        assert seen == [dir_dataset_path]

    def test_read_returns_dtype(self, source):
        assert isinstance(source.read(), dict)

    def test_read_contents(self, source, dir_dataset_path):
        data = source.read()
        expected = np.random.default_rng(seed=42).normal(
            loc=0.0, scale=1.0, size=blob_shape[1:])
        assert len(data) == blob_shape[0]
        assert set(data) == {os.path.splitext(f)[0]
                             for f in os.listdir(dir_dataset_path)}
        for batch in data.values():
            assert batch.shape == blob_shape[1:]
            np.testing.assert_array_equal(batch, expected)

    def test_custom_dtype(self, dir_dataset_path):
        cfg = DirectorySourceConfig(loc=dir_dataset_path, load_fn=load_dir,
                                    dtype=list)
        data = DirectorySource(cfg).read()
        assert isinstance(data, list)
        assert len(data) == blob_shape[0]

    def test_missing_directory(self, tmp_path):
        src = DirectorySource(DirectorySourceConfig(loc=tmp_path / 'missing',
                                                    load_fn=load_dir))
        with pytest.raises(FileNotFoundError):
            src.read()


class TestDirectorySourceLength:
    def test_length(self, source):
        assert len(source) == blob_shape[0]

    @pytest.mark.parametrize('num_batches', [1, 3, 16])
    def test_length_matches_num_batches(self, tmp_path, num_batches):
        path = gen_random_directory_source(tmp_path / 'blobs', num_batches=num_batches,
                                           shape=blob_shape[1:], ext_='.npy')
        src = DirectorySource(DirectorySourceConfig(loc=path, load_fn=load_dir))
        assert len(src) == num_batches

    def test_length_does_not_read(self, dir_dataset_path):
        loader = CountingLoader()
        src = DirectorySource(DirectorySourceConfig(loc=dir_dataset_path, load_fn=loader))
        len(src)
        assert loader.calls == 0

    def test_empty_directory(self, tmp_path):
        src = DirectorySource(DirectorySourceConfig(loc=tmp_path, load_fn=load_dir))
        assert len(src) == 0

    @pytest.mark.parametrize('batch_size', [1, 8, 32])
    def test_batched_length(self, dir_dataset_path, batch_size):
        cfg = DirectorySourceConfig(loc=dir_dataset_path, load_fn=load_dir,
                                    batch_size=batch_size)
        src = DirectorySource(cfg, batched=True)
        assert len(src) == blob_shape[0] * batch_size

    def test_batch_size_ignored_when_not_batched(self, dir_dataset_path):
        cfg = DirectorySourceConfig(loc=dir_dataset_path, load_fn=load_dir,
                                    batch_size=8)
        assert len(DirectorySource(cfg)) == blob_shape[0]

    def test_batched_without_batch_size(self, dir_dataset_path):
        src = DirectorySource(DirectorySourceConfig(loc=dir_dataset_path,
                                                    load_fn=load_dir), batched=True)
        with pytest.raises(TypeError):
            len(src)

    def test_missing_directory(self, tmp_path):
        src = DirectorySource(DirectorySourceConfig(loc=tmp_path / 'missing',
                                                    load_fn=load_dir))
        with pytest.raises(FileNotFoundError):
            len(src)

    @pytest.mark.parametrize('ext_', ['.npz', '.npy', '.bin'])
    def test_generator_writes_one_file_per_batch(self, tmp_path, ext_):
        path = gen_random_directory_source(tmp_path / 'blobs', num_batches=blob_shape[0],
                                           shape=blob_shape[1:], ext_=ext_)
        assert all(f.endswith(ext_) for f in os.listdir(path))
        src = DirectorySource(DirectorySourceConfig(loc=path, load_fn=load_dir))
        assert len(src) == blob_shape[0]
