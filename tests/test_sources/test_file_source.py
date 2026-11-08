import numpy as np
import pytest
from pydantic import ValidationError
from dragon_torch.data.sources.base import Source, source_factory
from dragon_torch.data.sources.disk import FileSource, FileSourceConfig
from dragon_torch.utils import gen_random_file_source

blob_shape = (4, 10, 2)  # BATCH_SIZE, XDim, YDim


def load_blob(path, **kw):
    # np.save pickles the dict into a 0-d object array; unwrap it
    return np.load(path, **kw).item()


class CountingLoader:
    """load_fn wrapper that records how often the source hits disk."""

    def __init__(self, fn=load_blob):
        self.fn = fn
        self.calls = []

    def __call__(self, path, **kw):
        self.calls.append(kw)
        return self.fn(path, **kw)


@pytest.fixture
def file_dataset_path(tmp_path):
    return gen_random_file_source(tmp_path / 'blob', num_batches=blob_shape[0],
                                  shape=blob_shape[1:])


@pytest.fixture
def source(file_dataset_path):
    src = FileSource(FileSourceConfig(loc=file_dataset_path, load_fn=load_blob))
    src.set_read_option('allow_pickle', True)
    return src


class TestFileSourceConfig:
    def test_defaults(self, file_dataset_path):
        cfg = FileSourceConfig(loc=file_dataset_path, load_fn=load_blob)
        assert cfg.single_file == True
        assert cfg.kind == 'file'
        assert cfg.read_bytes == False

    def test_accepts_str_loc(self, file_dataset_path):
        cfg = FileSourceConfig(loc=str(file_dataset_path), load_fn=load_blob)
        assert str(cfg.loc) == str(file_dataset_path)

    def test_requires_loc(self):
        with pytest.raises(ValidationError):
            FileSourceConfig(load_fn=load_blob)

    def test_requires_callable_load_fn(self, file_dataset_path):
        with pytest.raises(ValidationError):
            FileSourceConfig(loc=file_dataset_path, load_fn='np.load')


class TestFileSourceInit:
    def test_is_source(self, source):
        assert isinstance(source, Source)

    def test_registered_under_kind(self):
        assert Source._registry['file'] is FileSource

    def test_source_factory(self, file_dataset_path):
        cfg = FileSourceConfig(loc=file_dataset_path, load_fn=load_blob)
        assert isinstance(source_factory(cfg), FileSource)

    def test_init_does_not_read(self, file_dataset_path):
        loader = CountingLoader()
        FileSource(FileSourceConfig(loc=file_dataset_path, load_fn=loader))
        assert loader.calls == []


class TestFileSourceRead:
    def test_read_returns_dtype(self, source):
        assert isinstance(source.read(), dict)

    def test_read_contents(self, source):
        data = source.read()
        expected = np.random.default_rng(seed=42).normal(
            loc=0.0, scale=1.0, size=blob_shape[1:])
        assert len(data) == blob_shape[0]
        for batch in data.values():
            assert batch.shape == blob_shape[1:]
            np.testing.assert_array_equal(batch, expected)

    def test_read_without_pickle_fails(self, file_dataset_path):
        src = FileSource(FileSourceConfig(loc=file_dataset_path, load_fn=load_blob))
        with pytest.raises(ValueError):
            src.read()

    def test_read_option_forwarded_to_load_fn(self, file_dataset_path):
        loader = CountingLoader()
        src = FileSource(FileSourceConfig(loc=file_dataset_path, load_fn=loader))
        src.set_read_option('allow_pickle', True)
        src.read()
        assert loader.calls == [{'allow_pickle': True}]

    def test_multiple_read_options_stack(self, file_dataset_path):
        loader = CountingLoader()
        src = FileSource(FileSourceConfig(loc=file_dataset_path, load_fn=loader))
        src.set_read_option('allow_pickle', True)
        src.set_read_option('mmap_mode', None)
        src.read()
        assert loader.calls == [{'allow_pickle': True, 'mmap_mode': None}]

    def test_custom_dtype(self, file_dataset_path):
        cfg = FileSourceConfig(loc=file_dataset_path, load_fn=load_blob,
                               dtype=list)
        src = FileSource(cfg)
        src.set_read_option('allow_pickle', True)
        data = src.read()
        assert isinstance(data, list)
        assert len(data) == blob_shape[0]

    def test_missing_file(self, tmp_path):
        src = FileSource(FileSourceConfig(loc=tmp_path / 'missing.npy',
                                          load_fn=load_blob))
        src.set_read_option('allow_pickle', True)
        with pytest.raises(FileNotFoundError):
            src.read()


class TestFileSourceLength:
    def test_length(self, source):
        assert len(source) == blob_shape[0]

    @pytest.mark.parametrize('num_batches', [1, 3, 16])
    def test_length_matches_num_batches(self, tmp_path, num_batches):
        path = gen_random_file_source(tmp_path / 'blob', num_batches=num_batches,
                                      shape=blob_shape[1:])
        src = FileSource(FileSourceConfig(loc=path, load_fn=load_blob))
        src.set_read_option('allow_pickle', True)
        assert len(src) == num_batches

    def test_length_is_cached(self, file_dataset_path):
        loader = CountingLoader()
        src = FileSource(FileSourceConfig(loc=file_dataset_path, load_fn=loader))
        src.set_read_option('allow_pickle', True)
        assert len(src) == len(src) == blob_shape[0]
        assert len(loader.calls) == 1

    def test_shape_of_dict_source_is_none(self, source):
        assert source.shape() is None

    def test_shape_sets_length(self, file_dataset_path):
        loader = CountingLoader()
        src = FileSource(FileSourceConfig(loc=file_dataset_path, load_fn=loader))
        src.set_read_option('allow_pickle', True)
        src.shape()
        assert len(src) == blob_shape[0]
        assert len(loader.calls) == 1
