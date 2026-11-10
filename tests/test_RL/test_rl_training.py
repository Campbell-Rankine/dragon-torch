import pytest
import torch
from dragon_torch.data.envs import DragonEnvConfig
from dragon_torch.data.RL import EpisodeSource, RolloutSource
from dragon_torch.loss.RL import A2CLoss, PPOLoss
from dragon_torch.models.base import _DragonModel, model_factory
from dragon_torch.models.RL import (A2CConfig, A2CModel, DragonRLConfig, DragonRLModel, PPOConfig,
                                    PPOModel, SingleAgentRLModel)
from dragon_torch.modules.RL import ActorCritic, PolicyOutput
from dragon_torch.trainers import DragonTrainer, DragonTrainerConfig


def cartpole(n_envs=4, seed=0):
    return DragonEnvConfig(env_id='CartPole-v1', n_envs=n_envs, seed=seed)


def build(tmp_path, source, loss=None, **cfg_kw):
    torch.manual_seed(0)
    col = source.collector
    cfg = DragonRLConfig(checkpoint_dir=tmp_path / 'ckpt', save_every=1000, **cfg_kw)
    return SingleAgentRLModel(ActorCritic(col.observation_space, col.action_space),
                              loss if loss is not None else PPOLoss(), cfg)


def adam(params):
    return torch.optim.Adam(params, lr=2.5e-3, eps=1e-5)


class TestDragonRLModel:
    def test_registered_and_interface(self, tmp_path):
        assert _DragonModel._registry['rl-model'] is SingleAgentRLModel
        assert DragonRLModel.__abstractmethods__ >= {'act', 'value'}  # abstract base
        assert SingleAgentRLModel.higher_is_better and not SingleAgentRLModel.__abstractmethods__
        src = RolloutSource(cartpole(1), n_steps=4, n_minibatches=1)
        cfg = DragonRLConfig(checkpoint_dir=tmp_path, save_every=1)
        module = ActorCritic(src.collector.observation_space, src.collector.action_space)
        assert type(model_factory(module, PPOLoss(), cfg)) is SingleAgentRLModel

    def test_algorithm_models(self, tmp_path):
        src = RolloutSource(cartpole(1), n_steps=4, n_minibatches=1)
        module = lambda: ActorCritic(src.collector.observation_space, src.collector.action_space)
        assert _DragonModel._registry['ppo-model'] is PPOModel
        assert _DragonModel._registry['a2c-model'] is A2CModel
        ppo = PPOModel(module(), PPOConfig(checkpoint_dir=tmp_path, save_every=1, clip_coef=0.1))
        assert isinstance(ppo, SingleAgentRLModel) and isinstance(ppo.loss_fn, PPOLoss)
        assert ppo.loss_fn.clip_coef == 0.1
        a2c = A2CModel(module(), A2CConfig(checkpoint_dir=tmp_path, save_every=1, ent_coef=0.01))
        assert isinstance(a2c.loss_fn, A2CLoss) and a2c.loss_fn.ent_coef == 0.01
        # the config's kind picks the class; the factory's loss_fn overrides the config's
        built = model_factory(module(), A2CLoss(vf_coef=0.9), A2CConfig(checkpoint_dir=tmp_path, save_every=1))
        assert type(built) is A2CModel and built.loss_fn.vf_coef == 0.9
        with pytest.raises(TypeError, match='PPOConfig'):
            PPOModel(module(), A2CConfig(checkpoint_dir=tmp_path, save_every=1))
        src.close()

    def test_act_and_value(self, tmp_path):
        src = RolloutSource(cartpole(), n_steps=4)
        model = build(tmp_path, src)
        obs = src.collector.obs.astype('float64')  # float64 is cast down
        out = model.act(obs)
        assert out.action.shape == (4,) and set(out.extras) == {'log_prob', 'value'}
        assert model.value(obs).shape == (4,)
        assert isinstance(model.forward(torch.as_tensor(obs).float()), PolicyOutput)

    def test_ppo_learns_cartpole(self, tmp_path):
        source = RolloutSource(cartpole(), n_steps=128)
        evaluator = EpisodeSource(cartpole(seed=100), n_episodes=8)
        trainer = DragonTrainer(build(tmp_path, source, total_timesteps=25_000), adam)
        before = trainer.evaluate(evaluator)
        history = trainer.fit(source)
        after = trainer.evaluate(evaluator)
        assert before < 50 and after > 200
        assert history[-1]['global_step'] >= 25_000
        assert {'policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_frac',
                'steps_per_sec'} <= set(history[-1])

    def test_a2c_runs_with_validation(self, tmp_path):
        source = RolloutSource(cartpole(), n_steps=8, update_epochs=1, n_minibatches=1)
        trainer = DragonTrainer(build(tmp_path, source, A2CLoss()), adam)
        history = trainer.fit(source, val_data=EpisodeSource(cartpole(1), n_episodes=1), epochs=3)
        assert len(history) == 3 and 'val_return' in history[0] and 'approx_kl' not in history[0]

    def test_callbacks_see_global_step_and_can_stop(self, tmp_path):
        source = RolloutSource(cartpole(2), n_steps=10, n_minibatches=1)  # 20 steps / epoch
        model = build(tmp_path, source, total_timesteps=200)
        steps = []

        def cb(m, record):
            steps.append(record['global_step'])
            return record['global_step'] >= 60

        history = DragonTrainer(model, adam).fit(source, callbacks=[cb])
        assert steps == [20, 40, 60] and len(history) == 3

    def test_continuous_actions(self, tmp_path):
        source = RolloutSource(DragonEnvConfig(env_id='Pendulum-v1', n_envs=2), n_steps=16)
        history = DragonTrainer(build(tmp_path, source), adam).fit(source, epochs=2)
        assert len(history) == 2

    def test_epochs_from_total_timesteps(self, tmp_path):
        source = RolloutSource(cartpole(2), n_steps=10, n_minibatches=1)  # 20 steps / epoch
        model = build(tmp_path, source, total_timesteps=50)
        assert len(model.fit(source, adam(model.parameters()))) == 3
        assert model.fit(source, adam(model.parameters())) == []  # budget spent

    def test_resume_restores_normalizer_and_budget(self, tmp_path):
        def run():
            source = RolloutSource(cartpole(2), n_steps=10, n_minibatches=1, normalize_obs=True)
            return source, build(tmp_path, source, total_timesteps=80)

        source, model = run()
        model.fit(source, adam(model.parameters()), epochs=2)
        model.save_checkpoint()
        saved_mean = source.collector.obs_normalizer.rms.mean.copy()

        source2, model2 = run()
        trainer = DragonTrainer(model2, adam, DragonTrainerConfig(resume=True))
        history = trainer.fit(source2)
        assert [h['epoch'] for h in history] == [3, 4]
        assert source2.global_step == 80
        assert source2.collector.obs_normalizer.rms.count > 40  # restored + new, not fresh
        assert not (source2.collector.obs_normalizer.rms.mean == saved_mean).all()

    def test_eval_uses_train_normalizer(self, tmp_path):
        source = RolloutSource(cartpole(2), n_steps=10, n_minibatches=1, normalize_obs=True)
        model = build(tmp_path, source)
        evaluator = EpisodeSource(cartpole(1), n_episodes=1)
        model.fit(source, adam(model.parameters()), val_data=evaluator, epochs=1)
        assert evaluator.collector.obs_normalizer is source.collector.obs_normalizer
        count = source.collector.obs_normalizer.rms.count
        model.evaluate(evaluator)
        assert source.collector.obs_normalizer.rms.count == count  # eval never updates stats


class TestSB3Adapter:
    def test_dragon_module_as_features_extractor(self):
        sb3 = pytest.importorskip('stable_baselines3')
        from dragon_torch.modules.base import DragonModule
        from dragon_torch.modules.external.sb3 import DragonFeaturesExtractor, sb3_policy_kwargs

        class Encoder(DragonModule):
            kind = 'test-sb3-encoder'

            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Tanh())

            def forward(self, X, **kw):
                return self.net(X)

        encoder = Encoder()
        model = sb3.PPO('MlpPolicy', 'CartPole-v1', n_steps=32, batch_size=16, n_epochs=1,
                        policy_kwargs=sb3_policy_kwargs(encoder, features_dim=16), seed=0)
        model.learn(total_timesteps=32)
        extractor = model.policy.features_extractor
        assert isinstance(extractor, DragonFeaturesExtractor) and extractor.module is encoder
