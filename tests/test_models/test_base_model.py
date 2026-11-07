import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader
from data.datasets.aggregator import Aggregator
from data.datasets.dragon import DragonDataset
from data.sources.disk import FileSourceConfig
from loss import DragonLossFunction
from models.base import _DragonModel, DragonModel, DragonModelConfig, model_factory
from modules import DragonModule

IN_DIM, OUT_DIM, BATCH, N_BATCHES = 3, 1, 8, 6
TRUE_W = torch.tensor([[1.5], [-2.0], [0.5]])


class Linear(DragonModule):
    kind = 'test-linear'

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(IN_DIM, OUT_DIM)

    def forward(self, X, **kw):
        return self.layer(X)


class Identity(DragonModule):
    kind = 'test-identity'

    def forward(self, X, **kw):
        return X


class SubModel(DragonModel):
    kind = 'test-sub-model'


class NoKindModel(DragonModel):
    pass


def make_batches(n_batches=N_BATCHES, seed=0, as_numpy=False):
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        X = torch.randn(BATCH, IN_DIM, generator=g)
        batch = {'X': X, 'Y': X @ TRUE_W}
        batches.append({k: v.numpy() for k, v in batch.items()} if as_numpy else batch)
    return batches


def file_dataset(path, batches):
    torch.save(batches, path)
    return DragonDataset('file', FileSourceConfig(loc=path, load_fn=torch.load, dtype=list))


@pytest.fixture
def cfg(tmp_path):
    return DragonModelConfig(checkpoint_dir=tmp_path / 'ckpt', save_every=2, max_epochs=4)


@pytest.fixture
def model(cfg):
    torch.manual_seed(0)
    return DragonModel(Linear(), torch.nn.MSELoss(), cfg)


@pytest.fixture
def dataset(tmp_path):
    return file_dataset(tmp_path / 'train.pt', make_batches())


@pytest.fixture
def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.1)


class TestDragonModelConfig:
    def test_defaults(self, tmp_path):
        cfg = DragonModelConfig(checkpoint_dir=tmp_path, save_every=1, max_epochs=1)
        assert cfg.kind == 'torch-model'
        assert cfg.device == 'cpu'
        assert (cfg.input_key, cfg.target_key) == ('X', 'Y')

    def test_accepts_str_checkpoint_dir(self, tmp_path):
        cfg = DragonModelConfig(checkpoint_dir=str(tmp_path), save_every=1, max_epochs=1)
        assert str(cfg.checkpoint_dir) == str(tmp_path)

    @pytest.mark.parametrize('missing', ['checkpoint_dir', 'save_every', 'max_epochs'])
    def test_required(self, tmp_path, missing):
        kw = dict(checkpoint_dir=tmp_path, save_every=1, max_epochs=1)
        kw.pop(missing)
        with pytest.raises(ValidationError):
            DragonModelConfig(**kw)

    @pytest.mark.parametrize('field', ['save_every', 'max_epochs'])
    def test_rejects_non_positive(self, tmp_path, field):
        kw = dict(checkpoint_dir=tmp_path, save_every=1, max_epochs=1)
        kw[field] = 0
        with pytest.raises(ValidationError):
            DragonModelConfig(**kw)


class TestRegistry:
    def test_base_registered(self):
        assert _DragonModel._registry['torch-model'] is DragonModel

    def test_subclass_kind_does_not_overwrite_parent(self):
        assert _DragonModel._registry['test-sub-model'] is SubModel
        assert _DragonModel._registry['torch-model'] is DragonModel

    def test_subclass_without_kind_registers_by_name(self):
        assert _DragonModel._registry['NoKindModel'] is NoKindModel
        assert _DragonModel._registry['torch-model'] is DragonModel

    def test_model_factory(self, tmp_path):
        cfg = DragonModelConfig(checkpoint_dir=tmp_path, save_every=1, max_epochs=1,
                                kind='test-sub-model')
        assert type(model_factory(Linear(), torch.nn.MSELoss(), cfg)) is SubModel

    def test_abstract_base(self):
        with pytest.raises(TypeError):
            _DragonModel()


class TestDragonModelInit:
    def test_wraps_callable_loss(self, model):
        assert isinstance(model.loss_fn, DragonLossFunction)

    def test_keeps_dragon_loss(self, cfg):
        loss = DragonLossFunction(torch.nn.MSELoss())
        assert DragonModel(Linear(), loss, cfg).loss_fn is loss

    def test_rejects_non_module(self, cfg):
        with pytest.raises(TypeError):
            DragonModel(lambda X: X, torch.nn.MSELoss(), cfg)

    def test_parameters_are_modules(self, model):
        assert len(list(model.parameters())) == 2

    def test_train_eval(self, model):
        assert model.eval().torch_module.training is False
        assert model.train().torch_module.training is True


class TestForwardAndLoss:
    def test_forward(self, model):
        assert model.forward(torch.randn(BATCH, IN_DIM)).shape == (BATCH, OUT_DIM)

    def test_call_is_forward(self, model):
        X = torch.randn(BATCH, IN_DIM)
        torch.testing.assert_close(model(X), model.forward(X))

    def test_forward_runs_hooks(self, model):
        seen = []
        model.torch_module.register_forward_hook(lambda *a: seen.append(True))
        model(torch.randn(BATCH, IN_DIM))
        assert seen == [True]

    def test_loss(self, model):
        Y = torch.ones(BATCH, OUT_DIM)
        assert model.loss(Y, Y).item() == 0.0

    def test_unsupervised_loss(self, cfg):
        model = DragonModel(Identity(), lambda Y_hat: Y_hat.pow(2).mean(), cfg)
        assert model.loss(torch.full((2, 2), 2.0)).item() == 4.0

    def test_penalize(self, model):
        loss = torch.tensor(1.0)
        assert model.penalize(loss, lambda l, w: l + w, 0.5).item() == 1.5

    def test_penalize_rejects_non_tensor(self, model):
        with pytest.raises(ValueError):
            model.penalize(torch.tensor(1.0), lambda l: 1.0)


class TestUnpackBatch:
    def test_dict(self, model):
        X, Y = torch.zeros(2, IN_DIM), torch.ones(2, OUT_DIM)
        out_X, out_Y = model.unpack_batch({'X': X, 'Y': Y, 'extra': 1})
        assert out_X is X and out_Y is Y

    def test_dict_custom_keys(self, tmp_path):
        # semasia items look like {'embeddings': ..., 'labels': ...}
        cfg = DragonModelConfig(checkpoint_dir=tmp_path, save_every=1, max_epochs=1,
                                input_key='embeddings', target_key='labels')
        model = DragonModel(Linear(), torch.nn.MSELoss(), cfg)
        X, Y = model.unpack_batch({'embeddings': torch.zeros(2, 3), 'labels': torch.ones(2)})
        assert X.shape == (2, 3) and Y.shape == (2,)

    def test_dict_without_target(self, model):
        _, Y = model.unpack_batch({'X': torch.zeros(2, IN_DIM)})
        assert Y is None

    def test_dict_missing_input(self, model):
        with pytest.raises(KeyError, match='X'):
            model.unpack_batch({'Y': torch.zeros(2)})

    def test_tuple(self, model):
        X, Y = model.unpack_batch((torch.zeros(2), torch.ones(2)))
        assert X.sum() == 0 and Y.sum() == 2

    def test_bare_tensor(self, model):
        X, Y = model.unpack_batch(torch.zeros(2))
        assert X.shape == (2,) and Y is None

    def test_numpy_converted(self, model):
        X, Y = model.unpack_batch({'X': np.zeros((2, 3)), 'Y': np.ones((2, 1))})
        assert isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor)

    def test_bad_sequence(self, model):
        with pytest.raises(ValueError):
            model.unpack_batch((1, 2, 3))


class TestWithDragonDataset:
    def test_iter_batches_indexes_dataset(self, model, dataset):
        # iter(DragonDataset) hits HF Dataset.__iter__ and fails; the model must index
        batches = list(model.iter_batches(dataset))
        assert len(batches) == N_BATCHES
        torch.testing.assert_close(batches[0]['X'], make_batches()[0]['X'])

    def test_training_step_updates_params(self, model, dataset, optimizer):
        before = [p.clone() for p in model.parameters()]
        loss = model.training_step(dataset[0], optimizer)
        assert isinstance(loss, float)
        assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))

    def test_evaluate_does_not_train(self, model, dataset):
        before = [p.clone() for p in model.parameters()]
        loss = model.evaluate(dataset)
        assert loss > 0
        assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
        assert model.torch_module.training is False

    def test_fit_reduces_loss(self, model, dataset, optimizer, tmp_path):
        val = file_dataset(tmp_path / 'val.pt', make_batches(2, seed=1))
        start = model.evaluate(val)
        history = model.fit(dataset, optimizer, val_data=val)
        assert [h['epoch'] for h in history] == [1, 2, 3, 4]
        assert history[-1]['val_loss'] < start
        assert history[-1]['train_loss'] < history[0]['train_loss']

    def test_fit_numpy_dataset(self, model, optimizer, tmp_path):
        path = tmp_path / 'np.npy'
        np.save(path, np.array(make_batches(as_numpy=True), dtype=object))
        cfg = FileSourceConfig(loc=path, load_fn=lambda p: np.load(p, allow_pickle=True),
                               dtype=list)
        history = model.fit(DragonDataset('file', cfg), optimizer, epochs=1)
        assert len(history) == 1

    def test_fit_dataloader(self, model, dataset, optimizer):
        history = model.fit(DataLoader(dataset, batch_size=None), optimizer, epochs=1)
        assert len(history) == 1

    def test_fit_aggregator(self, model, optimizer, tmp_path):
        rows = lambda seed: [{'X': b['X'][i], 'Y': b['Y'][i]}
                             for b in make_batches(2, seed=seed) for i in range(BATCH)]
        a = file_dataset(tmp_path / 'a.pt', rows(0))
        b = file_dataset(tmp_path / 'b.pt', rows(1))

        def collate(parts):
            samples = [s for part in parts for s in part]
            return {k: torch.stack([s[k] for s in samples]) for k in ('X', 'Y')}

        agg = Aggregator([a, b], batch_size=4, weights={0: 0.5, 1: 0.5}, seed=0,
                         collate_fn=collate)
        history = model.fit(agg, optimizer, epochs=2)
        assert len(history) == 2 and agg.epoch == 1

    def test_fit_empty_data(self, model, optimizer):
        with pytest.raises(ValueError):
            model.fit([], optimizer, epochs=1)

    def test_non_scalar_loss_rejected(self, cfg, dataset):
        model = DragonModel(Linear(), torch.nn.MSELoss(reduction='none'), cfg)
        with pytest.raises(ValueError, match='scalar'):
            model.evaluate(dataset)


class TestCheckpoints:
    def test_fit_saves_every(self, model, dataset, optimizer, cfg):
        model.fit(dataset, optimizer)
        saved = sorted(p.name for p in cfg.checkpoint_dir.iterdir())
        assert saved == ['torch-model-epoch-2.pt', 'torch-model-epoch-4.pt']

    def test_fit_saves_final_epoch(self, model, dataset, optimizer, cfg):
        model.fit(dataset, optimizer, epochs=3)
        saved = sorted(p.name for p in cfg.checkpoint_dir.iterdir())
        assert saved == ['torch-model-epoch-2.pt', 'torch-model-epoch-3.pt']

    def test_fit_resumes_epoch_count(self, model, dataset, optimizer):
        model.fit(dataset, optimizer, epochs=1)
        assert model.fit(dataset, optimizer, epochs=1)[0]['epoch'] == 2

    def test_round_trip(self, model, dataset, optimizer, cfg):
        model.fit(dataset, optimizer, epochs=2)
        fresh = DragonModel(Linear(), torch.nn.MSELoss(), cfg)
        fresh_opt = torch.optim.SGD(fresh.parameters(), lr=0.1)
        assert fresh.load_checkpoint(model.checkpoint_path(), fresh_opt) == 2
        for a, b in zip(model.parameters(), fresh.parameters()):
            torch.testing.assert_close(a, b)
        assert fresh_opt.state_dict() == optimizer.state_dict()

    def test_save_without_optimizer(self, model, cfg):
        path = model.save_checkpoint()
        assert path.exists()
        fresh = DragonModel(Linear(), torch.nn.MSELoss(), cfg)
        assert fresh.load_checkpoint(path) == 0
