import pytest
import torch
from models.base import DragonModel, DragonModelConfig
from modules import DragonModule
from trainers import DragonTrainer, DragonTrainerConfig

IN_DIM = 3
TRUE_W = torch.tensor([[1.5], [-2.0], [0.5]])


class TrainerLinear(DragonModule):
    kind = 'test-trainer-linear'

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(IN_DIM, 1)

    def forward(self, X, **kw):
        return self.layer(X)


def batches(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        X = torch.randn(8, IN_DIM, generator=g)
        out.append({'X': X, 'Y': X @ TRUE_W})
    return out


@pytest.fixture
def cfg(tmp_path):
    return DragonModelConfig(checkpoint_dir=tmp_path / 'ckpt', save_every=1, max_epochs=3)


def make_model(cfg):
    torch.manual_seed(0)
    return DragonModel(TrainerLinear(), torch.nn.MSELoss(), cfg)


def sgd(params):
    return torch.optim.SGD(params, lr=0.1)


class TestDragonTrainer:
    def test_rejects_non_model(self):
        with pytest.raises(TypeError):
            DragonTrainer(TrainerLinear(), sgd)

    def test_builds_optimizer_from_factory(self, cfg):
        model = make_model(cfg)
        trainer = DragonTrainer(model, sgd)
        assert isinstance(trainer.optimizer, torch.optim.SGD)
        assert trainer.optimizer.param_groups[0]['params'][0] is next(model.parameters())

    def test_accepts_optimizer_instance(self, cfg):
        model = make_model(cfg)
        opt = sgd(model.parameters())
        assert DragonTrainer(model, opt).optimizer is opt

    def test_fit_uses_model_default_epochs(self, cfg):
        trainer = DragonTrainer(make_model(cfg), sgd)
        history = trainer.fit(batches(), val_data=batches(2, seed=1))
        assert [h['epoch'] for h in history] == [1, 2, 3]
        assert history[-1]['val_loss'] < history[0]['val_loss']

    def test_config_epochs_and_history_accumulates(self, cfg):
        trainer = DragonTrainer(make_model(cfg), sgd, DragonTrainerConfig(epochs=1))
        trainer.fit(batches())
        trainer.fit(batches(), epochs=2)
        assert [h['epoch'] for h in trainer.history] == [1, 2, 3]

    def test_evaluate_and_save(self, cfg):
        trainer = DragonTrainer(make_model(cfg), sgd)
        assert trainer.evaluate(batches()) > 0
        assert trainer.save_checkpoint().exists()

    def test_resume_from_latest(self, cfg):
        first = DragonTrainer(make_model(cfg), sgd)
        first.fit(batches(), epochs=2)

        second = DragonTrainer(make_model(cfg), sgd, DragonTrainerConfig(resume=True, epochs=1))
        history = second.fit(batches())
        assert history[0]['epoch'] == 3
        assert second.optimizer.state_dict()['param_groups'] == first.optimizer.state_dict()['param_groups']

    def test_resume_with_no_checkpoint(self, cfg):
        trainer = DragonTrainer(make_model(cfg), sgd, DragonTrainerConfig(resume=True))
        assert trainer.resume() is None
        assert trainer.fit(batches(), epochs=1)[0]['epoch'] == 1

    def test_higher_is_better_passthrough(self, cfg):
        assert DragonTrainer(make_model(cfg), sgd).higher_is_better is False

    def test_resume_restores_source_state_before_fit(self, cfg):
        class CountingSource:
            def __init__(self):
                self.served = 0

            def __iter__(self):
                for b in batches():
                    self.served += 1
                    yield b

            def state_dict(self):
                return {'served': self.served}

            def load_state_dict(self, state):
                self.served = state['served']

        DragonTrainer(make_model(cfg), sgd).fit(CountingSource(), epochs=2)
        source = CountingSource()
        DragonTrainer(make_model(cfg), sgd, DragonTrainerConfig(resume=True)).fit(source, epochs=1)
        assert source.served == 18  # 12 restored + 6 from the resumed epoch
