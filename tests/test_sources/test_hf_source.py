import pytest
import requests
from datasets import IterableDataset
from pydantic import ValidationError
from dragon_torch.common.defaults import identity
from dragon_torch.data.sources.base import Source, source_factory
from dragon_torch.data.sources.hf import HuggingFaceSource, HFSourceConfig
from tests.conftest import SEMASIA_CELEBA_REPO, SEMASIA_CELEBA_MODEL, SEMASIA_CELEBA_EMBED_DIM


def make_cfg(download_path, **overrides):
    kw = dict(name='semasia-celeba', repository=SEMASIA_CELEBA_REPO,
              options=[SEMASIA_CELEBA_MODEL], device='cpu', dtype='float32',
              split='test', download_path=download_path)
    kw.update(overrides)
    return HFSourceConfig(**kw)


@pytest.fixture
def cfg(tmp_path):
    return make_cfg(tmp_path)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def size_payload(num_rows):
    # shape of https://datasets-server.huggingface.co/size?dataset=...
    return {'size': {
        'dataset': {'dataset': SEMASIA_CELEBA_REPO, 'num_rows': sum(num_rows.values())},
        'splits': [{'dataset': SEMASIA_CELEBA_REPO, 'config': config, 'split': split,
                    'num_rows': rows} for (config, split), rows in num_rows.items()],
    }, 'pending': [], 'failed': [], 'partial': False}


@pytest.fixture
def fake_size_api(monkeypatch):
    """Patch requests.get to serve a size response and record each call."""
    calls = []

    def install(num_rows):
        def fake_get(url, **kw):
            calls.append((url, kw))
            return FakeResponse(size_payload(num_rows))
        monkeypatch.setattr(requests, 'get', fake_get)
        return calls
    return install


class TestHFSourceConfig:
    def test_defaults(self, cfg):
        assert cfg.kind == 'hugging-face'
        assert cfg.bearer is None
        assert cfg.stream_from_hf == True
        assert cfg.keep_dataset_in_memory == False
        assert cfg.post_process is identity

    @pytest.mark.parametrize('field', ['name', 'repository', 'options', 'device',
                                       'dtype', 'split', 'download_path'])
    def test_required_fields(self, tmp_path, field):
        kw = make_cfg(tmp_path).model_dump()
        kw.pop(field)
        with pytest.raises(ValidationError):
            HFSourceConfig(**kw)

    @pytest.mark.parametrize('field,value', [('device', 'tpu'), ('dtype', 'complex64'),
                                             ('split', 'holdout')])
    def test_rejects_invalid_literals(self, tmp_path, field, value):
        with pytest.raises(ValidationError):
            make_cfg(tmp_path, **{field: value})

    @pytest.mark.parametrize('split', ['train', 'test', 'valid'])
    def test_accepts_splits(self, tmp_path, split):
        assert make_cfg(tmp_path, split=split).split == split


class TestHuggingFaceSourceInit:
    def test_is_source(self, cfg):
        assert isinstance(HuggingFaceSource(cfg), Source)

    def test_registered_under_kind(self):
        assert Source._registry['hugging-face'] is HuggingFaceSource

    def test_source_factory(self, cfg):
        assert isinstance(source_factory(cfg), HuggingFaceSource)

    def test_init_is_offline(self, cfg, monkeypatch):
        def no_network(*a, **kw):
            raise AssertionError('init should not touch the network')
        monkeypatch.setattr(requests, 'get', no_network)
        monkeypatch.setattr('datasets.load_dataset', no_network)
        assert HuggingFaceSource(cfg).cfg is cfg


class TestHuggingFaceSourceLength:
    def test_bearer_sent_as_auth_header(self, tmp_path, fake_size_api):
        calls = fake_size_api({(SEMASIA_CELEBA_MODEL, 'test'): 10})
        src = HuggingFaceSource(make_cfg(tmp_path, bearer='Bearer hf_token'))
        len(src)
        assert calls[0][1] == {'headers': {'Authorization': 'Bearer hf_token'}}

    def test_no_auth_header_without_bearer(self, cfg, fake_size_api):
        calls = fake_size_api({(SEMASIA_CELEBA_MODEL, 'test'): 10})
        len(HuggingFaceSource(cfg))
        assert calls[0][1] == {}

    def test_queries_repository(self, cfg, fake_size_api):
        calls = fake_size_api({(SEMASIA_CELEBA_MODEL, 'test'): 10})
        len(HuggingFaceSource(cfg))
        assert f'dataset={SEMASIA_CELEBA_REPO}' in calls[0][0]

    def test_length_for_config_and_split(self, cfg, fake_size_api):
        fake_size_api({(SEMASIA_CELEBA_MODEL, 'train'): 100,
                       (SEMASIA_CELEBA_MODEL, 'test'): 10,
                       ('other_model', 'test'): 99})
        assert len(HuggingFaceSource(cfg)) == 10

    def test_default_config_without_options(self, tmp_path, fake_size_api):
        fake_size_api({('default', 'test'): 7})
        assert len(HuggingFaceSource(make_cfg(tmp_path, options=[]))) == 7

    def test_missing_split_raises(self, cfg, fake_size_api):
        fake_size_api({(SEMASIA_CELEBA_MODEL, 'train'): 100})
        with pytest.raises(KeyError, match=SEMASIA_CELEBA_MODEL):
            len(HuggingFaceSource(cfg))


@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    cfg = make_cfg(tmp_path_factory.mktemp('hf_cache'))
    return HuggingFaceSource(cfg).read()


@pytest.fixture(scope='module')
def first_row(dataset):
    return next(iter(dataset))


@pytest.mark.network
@pytest.mark.usefixtures('hf_hub_online')
class TestHuggingFaceSourceRead:
    def test_streams_iterable_dataset(self, dataset):
        assert isinstance(dataset, IterableDataset)

    def test_columns(self, dataset):
        assert {'id', 'celeb_id', 'model_name', 'embedding'} <= set(dataset.features)

    def test_row_is_from_requested_model(self, first_row):
        assert first_row['model_name'] == SEMASIA_CELEBA_MODEL

    def test_embedding_dim(self, first_row):
        assert len(first_row['embedding']) == SEMASIA_CELEBA_EMBED_DIM

    def test_take(self, dataset):
        assert len(list(dataset.take(5))) == 5

    def test_unknown_model_config(self, tmp_path):
        src = HuggingFaceSource(make_cfg(tmp_path, options=['not-a-model']))
        with pytest.raises(ValueError, match='not-a-model'):
            src.read()
