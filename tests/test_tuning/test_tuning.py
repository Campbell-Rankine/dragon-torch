import io
import json
import math
import pytest

optuna = pytest.importorskip('optuna', reason="tuning needs the 'tune' extra")
import torch
from dragon_torch.models.base import DragonModel, DragonModelConfig
from dragon_torch.models.RL import DragonRLModel
from dragon_torch.modules import DragonModule
from dragon_torch.utils.tuning import (DragonObjective, OptunaReporter, TrialSpec, TuneConfig, make_storage,
                          read_history, run_study)
from dragon_torch.utils.tuning.config import JournalStorage
from dragon_torch.utils.tuning.monitor import export_csv, main, summarize, trial_rows
from dragon_torch.utils.tuning.objective import final_value

optuna.logging.set_verbosity(optuna.logging.WARNING)
IN_DIM = 3
TRUE_W = torch.tensor([[1.5], [-2.0], [0.5]])


class TuneLinear(DragonModule):
    kind = 'test-tune-linear'

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(IN_DIM, 1)

    def forward(self, X, **kw):
        return self.layer(X)


def batches(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        X = torch.randn(8, IN_DIM, generator=g)
        out.append({'X': X, 'Y': X @ TRUE_W})
    return out


class ClosableData(list):
    closed = False

    def close(self):
        self.closed = True


def suggest(trial):
    return {'lr': trial.suggest_float('lr', 1e-3, 1e-1, log=True)}


def make_build(epochs=3, callbacks=(), data_log=None):
    def build(params, trial_dir):
        torch.manual_seed(0)
        cfg = DragonModelConfig(checkpoint_dir=trial_dir / 'ckpt', save_every=100, max_epochs=epochs)
        train = ClosableData(batches())
        if data_log is not None:
            data_log.append(train)
        return TrialSpec(model=DragonModel(TuneLinear(), torch.nn.MSELoss(), cfg),
                         optimizer=lambda p: torch.optim.SGD(p, lr=params['lr']),
                         train_data=train, val_data=batches(2, seed=1), callbacks=callbacks)
    return build


def tune_cfg(tmp_path, **kw):
    kw.setdefault('study_name', 'test')
    kw.setdefault('run_dir', tmp_path / 'runs')
    kw.setdefault('n_trials', 3)
    kw.setdefault('seed', 0)
    return TuneConfig.for_model(DragonModel, **kw)


class AlwaysPrune(optuna.pruners.BasePruner):
    def prune(self, study, trial):
        return True


class TestTuneConfig:
    def test_for_model_reads_metric_and_direction(self, tmp_path):
        assert tune_cfg(tmp_path).metrics == ['val_loss']
        assert tune_cfg(tmp_path).directions == ['minimize']
        rl = TuneConfig.for_model(DragonRLModel, study_name='rl')
        assert rl.metrics == ['val_return'] and rl.directions == ['maximize']

    def test_validation(self):
        with pytest.raises(ValueError, match='directions'):
            TuneConfig(study_name='x', metrics=['a', 'b'], directions=['minimize'])
        with pytest.raises(ValueError, match='duplicate'):
            TuneConfig(study_name='x', metrics=['a', 'a'], directions=['minimize'] * 2)
        with pytest.raises(ValueError, match='cmaes'):
            TuneConfig(study_name='x', metrics=['a', 'b'], directions=['minimize'] * 2,
                       sampler='cmaes')

    def test_multi_objective_never_prunes(self):
        cfg = TuneConfig(study_name='x', metrics=['a', 'b'], directions=['minimize'] * 2)
        assert cfg.multi_objective and not cfg.can_prune
        assert isinstance(cfg.make_pruner(), optuna.pruners.NopPruner)

    def test_make_storage(self, tmp_path):
        assert make_storage(None) is None
        url = f"sqlite:///{tmp_path / 'sub' / 'o.db'}"
        assert make_storage(url) == url and (tmp_path / 'sub').is_dir()
        assert isinstance(make_storage(f"journal:{tmp_path / 'j' / 'o.log'}"), JournalStorage)


class TestFinalValue:
    @pytest.mark.parametrize('final,direction,window,expected', [
        ('last', 'minimize', 1, 1.0),
        ('mean_last', 'maximize', 2, 2.0),
        ('best', 'minimize', 1, 1.0),
        ('best', 'maximize', 1, 4.0),
    ])
    def test_reductions(self, final, direction, window, expected):
        assert final_value([2.0, 4.0, 3.0, 1.0], direction, final, window) == expected


class TestRunStudy:
    def test_single_objective(self, tmp_path):
        cfg = tune_cfg(tmp_path, storage=f"sqlite:///{tmp_path / 'o.db'}")
        data = []
        study = run_study(cfg, suggest, make_build(data_log=data))
        trials = study.get_trials()
        assert [t.state for t in trials] == [optuna.trial.TrialState.COMPLETE] * 3
        assert all(d.closed for d in data)

        t = study.best_trial
        history = read_history(f"{t.user_attrs['trial_dir']}/history.jsonl")
        assert [r['epoch'] for r in history] == [1, 2, 3]
        assert t.value == history[-1]['val_loss'] == t.user_attrs['last_record']['val_loss']
        assert t.intermediate_values == {r['epoch']: r['val_loss'] for r in history}

        best = json.loads((cfg.study_dir / 'best.json').read_text())
        assert best['best'][0]['number'] == t.number and best['directions'] == ['minimize']

    def test_workers_share_storage(self, tmp_path):
        """A second run of the same study joins it (another worker, or a restart)."""
        for storage in (f"sqlite:///{tmp_path / 'o.db'}", f"journal:{tmp_path / 'o.log'}"):
            cfg = tune_cfg(tmp_path, storage=storage, n_trials=2)
            run_study(cfg, suggest, make_build(epochs=1))
            study = run_study(cfg.model_copy(update={'seed': 1}), suggest, make_build(epochs=1))
            assert len(study.trials) == 4
            assert len({t.user_attrs['trial_dir'] for t in study.trials}) == 4

    def test_multi_objective(self, tmp_path):
        cfg = tune_cfg(tmp_path, metrics=['val_loss', 'train_loss'],
                       directions=['minimize', 'minimize'], final='mean_last', final_window=2)
        study = run_study(cfg, suggest, make_build())
        t = study.trials[0]
        history = read_history(f"{t.user_attrs['trial_dir']}/history.jsonl")
        expected = [sum(r[m] for r in history[-2:]) / 2 for m in cfg.metrics]
        assert t.values == pytest.approx(expected)
        assert t.intermediate_values == {}  # no trial.report in multi-objective studies
        assert json.loads((cfg.study_dir / 'best.json').read_text())['best']

    def test_pruning(self, tmp_path):
        cfg = tune_cfg(tmp_path)
        study = optuna.create_study(pruner=AlwaysPrune())
        data = []
        study.optimize(DragonObjective(cfg, suggest, make_build(epochs=5, data_log=data)), n_trials=2)
        assert [t.state for t in study.trials] == [optuna.trial.TrialState.PRUNED] * 2
        assert all(t.last_step == 1 for t in study.trials)
        assert all(d.closed for d in data)
        assert len(read_history(f"{study.trials[0].user_attrs['trial_dir']}/history.jsonl")) == 1

    def test_non_finite_metric_fails_trial_and_study_continues(self, tmp_path):
        def diverge(model, record):
            if record['epoch'] == 2:
                record['val_loss'] = float('nan')

        cfg = tune_cfg(tmp_path, n_trials=2)
        study = run_study(cfg, suggest, make_build(epochs=5, callbacks=[diverge]))
        assert [t.state for t in study.trials] == [optuna.trial.TrialState.FAIL] * 2
        assert study.trials[0].user_attrs['epochs'] == 2  # reporter stopped training
        assert json.loads((cfg.study_dir / 'best.json').read_text())['best'] == []

    def test_missing_metric_is_an_error(self, tmp_path):
        cfg = tune_cfg(tmp_path, metrics=['nope'], directions=['minimize'], n_trials=1)
        with pytest.raises(ValueError, match="'nope' never appeared"):
            run_study(cfg, suggest, make_build(epochs=1))


class TestOptunaReporter:
    def test_skips_records_without_metric(self):
        trial = optuna.create_study().ask()
        reporter = OptunaReporter(trial, 'val_return', step_key='global_step')
        assert reporter(None, {'epoch': 1, 'global_step': 10}) is None
        assert reporter(None, {'epoch': 2, 'global_step': 20, 'val_return': 5.0}) is None
        assert trial.storage.get_trial(trial._trial_id).intermediate_values == {20: 5.0}

    def test_stops_on_non_finite(self):
        reporter = OptunaReporter(optuna.create_study().ask(), 'm', report=False)
        assert reporter(None, {'epoch': 1, 'm': math.inf}) is True
        assert reporter(None, {'epoch': 1, 'm': 1.0}) is None


class TestMonitor:
    @pytest.fixture
    def storage(self, tmp_path):
        url = f"sqlite:///{tmp_path / 'o.db'}"
        run_study(tune_cfg(tmp_path, storage=url, n_trials=2), suggest, make_build(epochs=1))
        return url

    def test_trial_rows(self, storage):
        rows = trial_rows(optuna.load_study(study_name='test', storage=storage))
        assert [r['state'] for r in rows] == ['COMPLETE'] * 2
        assert {'value', 'duration_s', 'last_step', 'param_lr'} <= set(rows[0])

    def test_summary(self, storage):
        out = io.StringIO()
        summarize(storage, 'test', out)
        text = out.getvalue()
        assert 'complete=2' in text and 'param_lr' in text and 'best:' in text
        out = io.StringIO()
        summarize(storage, None, out)
        assert out.getvalue().startswith('test: 2 trials')

    def test_export_and_cli(self, storage, tmp_path, capsys):
        path = export_csv(storage, 'test', tmp_path / 'out' / 'trials.csv')
        assert len(path.read_text().splitlines()) == 3
        main(['summary', '--storage', storage, '--study', 'test'])
        assert 'complete=2' in capsys.readouterr().out
