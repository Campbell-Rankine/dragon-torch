import pytest
import torch
from datasets import Dataset
from dragon_torch.data.datasets.dragon import _Dataset
from dragon_torch.data.datasets.hugging_face.semasia.config import extract_semasia_dataset
from dragon_torch.data.datasets.hugging_face.semasia.semasia_celeba import (
    SemasiaCeleba, SemasiaCelebaSingleModel, SemasiaCelebaSourceConfig)
from dragon_torch.data.sources.hf import HuggingFaceSource
from tests.conftest import SEMASIA_CELEBA_REPO, SEMASIA_CELEBA_MODEL, SEMASIA_CELEBA_EMBED_DIM


@pytest.fixture
def cfg(tmp_path):
    return SemasiaCelebaSourceConfig(split='test', download_path=tmp_path)


@pytest.fixture
def embeddings_ds():
    # stand-in for a downloaded split: 3 rows of 4-dim embeddings
    return Dataset.from_dict({
        'embedding': [[float(i)] * 4 for i in range(3)],
        'celeb_id': [10, 11, 12],
    }).with_format('torch')


class TestSemasiaCelebaSourceConfig:
    def test_defaults(self, cfg):
        assert cfg.name == 'semasia-celeba'
        assert cfg.repository == SEMASIA_CELEBA_REPO
        assert cfg.options == []
        assert cfg.device == 'cpu'
        assert cfg.dtype == 'float64'
        assert cfg.kind == 'hugging-face'
        assert cfg.post_process is extract_semasia_dataset

    def test_set_model(self, cfg):
        cfg.set_model(SEMASIA_CELEBA_MODEL)
        assert cfg.options == [SEMASIA_CELEBA_MODEL]

    def test_options_not_shared_between_instances(self, tmp_path):
        a = SemasiaCelebaSourceConfig(download_path=tmp_path)
        b = SemasiaCelebaSourceConfig(download_path=tmp_path)
        a.options.append(SEMASIA_CELEBA_MODEL)
        assert b.options == []


class TestExtractSemasiaDataset:
    def test_full_dataset(self, embeddings_ds):
        out = extract_semasia_dataset(embeddings_ds)
        assert set(out) == {'embeddings', 'labels'}
        assert out['embeddings'].shape == (3, 4)
        assert out['labels'].tolist() == [10, 11, 12]

    def test_slice(self, embeddings_ds):
        out = extract_semasia_dataset(embeddings_ds[1:3])
        torch.testing.assert_close(out['embeddings'],
                                   torch.tensor([[1.] * 4, [2.] * 4]))
        assert out['labels'].tolist() == [11, 12]

    def test_requires_torch_format(self, embeddings_ds):
        with pytest.raises(TypeError):
            extract_semasia_dataset(embeddings_ds.with_format(None))


class TestSemasiaCeleba:
    def test_registered(self):
        assert _Dataset._registry['hf-semasia-celeba'] is SemasiaCeleba
        assert _Dataset._registry['hf-semasia-celeba-single-model'] is SemasiaCelebaSingleModel

    def test_builds_hf_source(self, cfg):
        ds = SemasiaCeleba(cfg)
        assert isinstance(ds.source, HuggingFaceSource)
        assert ds.source.cfg is cfg
        assert ds.config is cfg

    def test_set_model(self, cfg):
        ds = SemasiaCeleba(cfg)
        ds.set_model(SEMASIA_CELEBA_MODEL)
        assert ds.config.options == [SEMASIA_CELEBA_MODEL]

    def test_set_model_twice_raises(self, cfg):
        ds = SemasiaCeleba(cfg)
        ds.set_model(SEMASIA_CELEBA_MODEL)
        with pytest.raises(AttributeError, match=SEMASIA_CELEBA_MODEL):
            ds.set_model('other_model')
        assert ds.config.options == [SEMASIA_CELEBA_MODEL]

    def test_post_process_uses_config(self, cfg, embeddings_ds):
        out = SemasiaCeleba(cfg).post_process(embeddings_ds)
        assert out['embeddings'].shape == (3, 4)

    def test_get_dataset_prefers_loaded_data(self, cfg, embeddings_ds, monkeypatch):
        ds = SemasiaCeleba(cfg)
        monkeypatch.setattr(ds.source, 'read', lambda: pytest.fail('should not read'))
        ds.set_data(embeddings_ds)
        assert ds.get_dataset() is embeddings_ds

    def test_get_dataset_falls_back_to_source(self, cfg, embeddings_ds, monkeypatch):
        ds = SemasiaCeleba(cfg)
        monkeypatch.setattr(ds.source, 'read', lambda: embeddings_ds)
        assert ds.get_dataset() is embeddings_ds

    def test_load(self, cfg, embeddings_ds, monkeypatch):
        ds = SemasiaCeleba(cfg)
        monkeypatch.setattr(ds.source, 'read', lambda: embeddings_ds)
        ds.load()
        assert ds.data is embeddings_ds

    def test_getitem(self, cfg, embeddings_ds):
        ds = SemasiaCeleba(cfg)
        ds.set_data(embeddings_ds)
        out = ds[1:3]
        assert out['embeddings'].shape == (2, 4)
        assert out['labels'].tolist() == [11, 12]

    def test_len_without_batch_size(self, cfg, monkeypatch):
        ds = SemasiaCeleba(cfg)
        monkeypatch.setattr(HuggingFaceSource, '__len__', lambda self: 10)
        assert ds.batch_size is None
        assert len(ds) == 10

    @pytest.mark.parametrize('batch_size,expected', [(1, 10), (3, 3), (5, 2), (20, 0)])
    def test_len_with_batch_size(self, cfg, monkeypatch, batch_size, expected):
        ds = SemasiaCeleba(cfg)
        ds.batch_size = batch_size
        monkeypatch.setattr(HuggingFaceSource, '__len__', lambda self: 10)
        assert len(ds) == expected


class TestSemasiaCelebaSingleModel:
    def test_default_model(self, cfg):
        ds = SemasiaCelebaSingleModel(cfg)
        assert ds.model_name == 'aimv2_1b_patch14_224.apple_pt'
        assert ds.config.options == ['aimv2_1b_patch14_224.apple_pt']

    def test_custom_model(self, cfg):
        ds = SemasiaCelebaSingleModel(cfg, model_name=SEMASIA_CELEBA_MODEL)
        assert ds.config.options == [SEMASIA_CELEBA_MODEL]
        assert ds.kind == 'hf-semasia-celeba-single-model'

    def test_model_locked_after_init(self, cfg):
        ds = SemasiaCelebaSingleModel(cfg, model_name=SEMASIA_CELEBA_MODEL)
        with pytest.raises(AttributeError):
            ds.set_model('other_model')

    @pytest.mark.network
    @pytest.mark.usefixtures('hf_hub_online')
    def test_streams_rows_for_model(self, cfg):
        ds = SemasiaCelebaSingleModel(cfg, model_name=SEMASIA_CELEBA_MODEL)
        rows = list(ds.get_dataset().take(2))
        assert len(rows) == 2
        for row in rows:
            assert row['model_name'] == SEMASIA_CELEBA_MODEL
            assert len(row['embedding']) == SEMASIA_CELEBA_EMBED_DIM
