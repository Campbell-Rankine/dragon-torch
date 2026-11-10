"""Multi-agent RL: env adapters and grouping, vectorisation, collection,
rollouts, loss / module / model, and end-to-end training with the
multi-agent trainers. Uses two small in-repo envs (no PettingZoo needed):

  MatchEnv      ParallelEnv. 'agent_0', 'agent_1' see a one-hot target in 3 and
                score 1 for picking it; 'scout_0' does the same in 2 but leaves
                after `scout_steps`. Every episode is truncated after `length`.
  TeamEnv       gymnasium env with Dict spaces: agents 'a' and 'b' see a shared
                bit, and the team reward (a scalar) counts how many copied it.
"""
import math
import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium import spaces
from dragon_torch.data.envs import (AgentGroups, EnvCheckError, GymMultiAgentEnv, MultiAgentEnvConfig,
                                    MultiAgentVectorEnv, is_parallel_env, make_multi_agent_env)
from dragon_torch.data.RL import (MultiAgentCollector, MultiAgentEpisodeSource,
                                  MultiAgentRolloutSource)
from dragon_torch.loss.RL import A2CLoss, MultiAgentLoss, PPOLoss
from dragon_torch.models.base import _DragonModel
from dragon_torch.models.RL import (DragonRLConfig, MultiAgentA2CConfig, MultiAgentA2CModel,
                                    MultiAgentPPOConfig, MultiAgentPPOModel, MultiAgentRLConfig,
                                    MultiAgentRLModel, PPOConfig)
from dragon_torch.modules.RL import ActorCritic, MultiAgentActorCritic, PolicyOutput
from dragon_torch.trainers.RL import (EvalConfig, MultiAgentA2CTrainer, MultiAgentA2CTrainerConfig,
                                      MultiAgentPPOTrainer, MultiAgentPPOTrainerConfig, PPOTrainer,
                                      RolloutConfig, apply_overrides)


class MatchEnv:
    metadata = {'name': 'match_v0'}

    def __init__(self, length: int = 8, scout_steps: int = 3):
        self.length, self.scout_steps = length, scout_steps
        self.possible_agents = ['agent_0', 'agent_1', 'scout_0']
        self.agents: list[str] = []
        self._n = {'agent_0': 3, 'agent_1': 3, 'scout_0': 2}
        self._rng = np.random.default_rng(0)
        self.closed = False

    def observation_space(self, agent):
        return spaces.Box(0.0, 1.0, (self._n[agent],), np.float32)

    def action_space(self, agent):
        return spaces.Discrete(self._n[agent])

    def _observe(self):
        self._target = {a: int(self._rng.integers(self._n[a])) for a in self.agents}
        return {a: np.eye(self._n[a], dtype=np.float32)[self._target[a]] for a in self.agents}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents, self.t = list(self.possible_agents), 0
        return self._observe(), {a: {} for a in self.agents}

    def step(self, actions):
        assert set(actions) == set(self.agents), (sorted(actions), self.agents)
        self.t += 1
        acting = list(self.agents)
        rewards = {a: float(int(actions[a]) == self._target[a]) for a in acting}
        terms = {a: a == 'scout_0' and self.t >= self.scout_steps for a in acting}
        truncs = {a: self.t >= self.length and not terms[a] for a in acting}
        obs = self._observe()
        self.agents = [a for a in acting if not (terms[a] or truncs[a])]
        return obs, rewards, terms, truncs, {a: {} for a in acting}

    def close(self):
        self.closed = True


class TeamEnv(gym.Env):
    def __init__(self, length: int = 5, reward_as: str = 'scalar'):
        self.length, self.reward_as = length, reward_as
        bit = spaces.Box(0.0, 1.0, (2,), np.float32)
        self.observation_space = spaces.Dict({'a': bit, 'b': bit})
        self.action_space = spaces.Dict({'a': spaces.Discrete(2), 'b': spaces.Discrete(2)})

    def _obs(self):
        self.bit = int(self.np_random.integers(2))
        one_hot = np.eye(2, dtype=np.float32)[self.bit]
        return {'a': one_hot, 'b': one_hot.copy()}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        hits = {k: float(int(action[k]) == self.bit) for k in ('a', 'b')}
        reward = {'scalar': sum(hits.values()), 'list': [hits['a'], hits['b']], 'dict': hits}[self.reward_as]
        return self._obs(), reward, self.t >= self.length, False, {}


class DoubleReward:
    """A ParallelEnv wrapper that doubles rewards (shaping)."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        obs, r, te, tr, info = self.env.step(actions)
        return obs, {a: 2 * v for a, v in r.items()}, te, tr, info


def match_cfg(n_envs=2, seed=0, **kw):
    return MultiAgentEnvConfig(env_fn=MatchEnv, n_envs=n_envs, seed=seed, **kw)


def team_cfg(n_envs=2, seed=0, **kw):
    return MultiAgentEnvConfig(env_fn=TeamEnv, n_envs=n_envs, seed=seed, **kw)


def random_actor(source_groups, value=True):
    """A multi-agent Actor taking uniform random actions."""
    from dragon_torch.data.RL import ActOutput

    class Random:
        def act(self, obs, deterministic=False):
            actions, logp, values = {}, {}, {}
            for g, o in obs.items():
                n = source_groups.action_spaces[g].n
                lead = o.shape[:2]
                actions[g] = torch.randint(n, lead)
                logp[g] = torch.full(lead, -math.log(n))
                values[g] = torch.zeros(lead)
            extras = {'log_prob': logp, **({'value': values} if value else {})}
            return ActOutput(actions, extras)

        def value(self, obs):
            return {g: torch.zeros(o.shape[:2]) for g, o in obs.items()}
    return Random()


def model_cfg(tmp_path, **kw):
    return MultiAgentRLConfig(checkpoint_dir=tmp_path / 'ckpt', save_every=1000, **kw)


# ------------------------------------------------------------------ env layer
class TestEnvLayer:
    def test_parallel_env_detection_and_adapter(self):
        assert is_parallel_env(MatchEnv())
        assert not is_parallel_env(TeamEnv())
        env = make_multi_agent_env(team_cfg())
        assert isinstance(env, GymMultiAgentEnv) and env.possible_agents == ['a', 'b']
        with pytest.raises(TypeError, match='ParallelEnv'):
            make_multi_agent_env(MultiAgentEnvConfig(env_fn=lambda: object()))

    def test_gym_adapter_requires_dict_spaces(self):
        with pytest.raises(TypeError, match='Dict'):
            GymMultiAgentEnv(gym.make('CartPole-v1'))

    @pytest.mark.parametrize('reward_as', ['scalar', 'list', 'dict'])
    def test_gym_adapter_reward_forms(self, reward_as):
        env = GymMultiAgentEnv(TeamEnv(length=2, reward_as=reward_as))
        obs, infos = env.reset(seed=0)
        assert set(obs) == {'a', 'b'} and set(infos) == {'a', 'b'}
        bit = int(obs['a'][1])
        obs, rewards, terms, truncs, _ = env.step({'a': bit, 'b': 1 - bit})
        expected = {'scalar': {'a': 1.0, 'b': 1.0}, 'list': {'a': 1.0, 'b': 0.0},
                    'dict': {'a': 1.0, 'b': 0.0}}[reward_as]
        assert rewards == expected
        assert terms == {'a': False, 'b': False} and env.agents == ['a', 'b']
        _, _, terms, _, _ = env.step({'a': 0, 'b': 0})
        assert terms == {'a': True, 'b': True} and env.agents == []

    def test_default_groups_by_prefix(self):
        groups = AgentGroups.from_env(MatchEnv())
        assert groups.agents_by_group == {'agent': ['agent_0', 'agent_1'], 'scout': ['scout_0']}
        assert groups.action_spaces['agent'] == spaces.Discrete(3)
        solo = AgentGroups.from_env(MatchEnv(), share_params=False)
        assert solo.names == ['agent_0', 'agent_1', 'scout_0']

    def test_group_validation(self):
        env = MatchEnv()
        with pytest.raises(ValueError, match='identical spaces'):
            AgentGroups.from_env(env, {'all': ['agent_0', 'agent_1', 'scout_0']})
        with pytest.raises(ValueError, match='missing'):
            AgentGroups.from_env(env, {'agent': ['agent_0', 'agent_1']})
        with pytest.raises(ValueError, match='identifier'):
            AgentGroups.from_env(env, {'a.b': ['agent_0', 'agent_1'], 's': ['scout_0']})
        with pytest.raises(ValueError, match='share_params'):
            match_cfg(groups={'x': ['agent_0']}, share_params=False)

    def test_vector_env_shapes_alive_and_autoreset(self):
        venv = MultiAgentVectorEnv(match_cfg(n_envs=2))
        obs, alive = venv.reset(seed=0)
        assert obs['agent'].shape == (2, 2, 3) and obs['scout'].shape == (2, 1, 2)
        assert alive['agent'].all() and alive['scout'].all()
        act = {'agent': np.zeros((2, 2), np.int64), 'scout': np.zeros((2, 1), np.int64)}
        for t in range(1, 9):
            res = venv.step(act)
            if t < 3:
                assert res.next_alive['scout'].all() and not res.terminated['scout'].any()
            if t == 3:  # scout leaves: terminated, no longer alive, obs padded
                assert res.terminated['scout'].all() and not res.next_alive['scout'].any()
                assert (res.next_obs['scout'] == 0).all()
            if 3 < t < 8:  # gone: no reward, no flags
                assert (res.reward['scout'] == 0).all() and not res.terminated['scout'].any()
            if t < 8:
                assert not res.env_done.any()
        # length reached: agents truncated, envs reset with everyone back
        assert res.env_done.all() and res.truncated['agent'].all()
        assert res.next_alive['agent'].all() and res.next_alive['scout'].all()
        venv.close()
        assert all(e.closed for e in venv.envs)

    def test_check_env(self):
        from dragon_torch.data.envs import check_multi_agent_env_config
        check_multi_agent_env_config(match_cfg())
        check_multi_agent_env_config(team_cfg())

        class BadObs(MatchEnv):
            def _observe(self):
                return {a: o * 5 for a, o in super()._observe().items()}
        with pytest.raises(EnvCheckError, match='outside its observation_space'):
            check_multi_agent_env_config(MultiAgentEnvConfig(env_fn=BadObs))


# ------------------------------------------------------------------ collection
class TestCollection:
    def test_episodes_team_and_group_returns(self):
        col = MultiAgentCollector(match_cfg(n_envs=2))
        actor = random_actor(col.groups)
        for _ in range(8):
            col.step(actor)
        eps = col.drain_episodes()
        assert len(eps) == 2 and all(e.length == 8 for e in eps)
        for e in eps:
            assert 0 <= e.group_returns['scout'] <= 3 and 0 <= e.group_returns['agent'] <= 8
            # sum over agents: 2 agents at their mean + the scout
            assert e.raw_return == pytest.approx(2 * e.group_returns['agent'] + e.group_returns['scout'])
        assert col.global_step == 16

    def test_team_return_mean(self):
        col = MultiAgentCollector(match_cfg(n_envs=1, team_return='mean'))
        actor = random_actor(col.groups)
        for _ in range(8):
            col.step(actor)
        (e,) = col.drain_episodes()
        assert e.raw_return == pytest.approx((2 * e.group_returns['agent'] + e.group_returns['scout']) / 3)

    def test_raw_vs_shaped(self):
        col = MultiAgentCollector(match_cfg(n_envs=1, wrappers=[DoubleReward]))
        assert col.shaped
        actor = random_actor(col.groups)
        for _ in range(8):
            col.step(actor)
        record = col.metrics()
        assert record['episode_return_shaped'] == pytest.approx(2 * record['episode_return'])
        assert 'episode_return/agent' in record and 'episode_return/scout' in record

    def test_gym_team_env(self):
        col = MultiAgentCollector(team_cfg(n_envs=2))
        assert col.groups.names == ['a', 'b']
        actor = random_actor(col.groups)
        for _ in range(5):
            col.step(actor)
        eps = col.drain_episodes()
        # every agent gets the scalar team reward; sum over the two agents
        assert len(eps) == 2 and all(e.raw_return == pytest.approx(2 * e.group_returns['a']) for e in eps)

    def test_normalisation_state_roundtrip(self):
        col = MultiAgentCollector(match_cfg(), normalize_obs=True, normalize_reward=True)
        actor = random_actor(col.groups)
        for _ in range(5):
            col.step(actor)
        state = col.state_dict()
        other = MultiAgentCollector(match_cfg(), normalize_obs=True, normalize_reward=True)
        other.load_state_dict(state)
        assert other.global_step == col.global_step
        for g in col.groups.names:
            assert np.allclose(other.obs_normalizers[g].rms.mean, col.obs_normalizers[g].rms.mean)

    def test_rollout_batches_only_live_agents(self):
        src = MultiAgentRolloutSource(match_cfg(n_envs=2), n_steps=8, update_epochs=1, n_minibatches=1)
        src.bind(random_actor(src.groups))
        (batch,) = list(src)
        # 8 steps x 2 envs: 2 agents always alive, the scout only for 3 steps
        assert batch['obs']['agent'].shape == (32, 3)
        assert batch['obs']['scout'].shape == (6, 2)
        targets = batch['targets']['scout']
        assert set(targets) == {'actions', 'log_prob', 'values', 'advantages', 'returns'}
        assert targets['actions'].shape == (6,)
        # values are 0, so returns are the discounted remaining scout rewards (<= 3)
        assert (targets['returns'] >= 0).all() and (targets['returns'] <= 3).all()

    def test_rollout_minibatches_and_state(self):
        src = MultiAgentRolloutSource(match_cfg(n_envs=2), n_steps=8, update_epochs=2, n_minibatches=4)
        src.bind(random_actor(src.groups))
        batches = list(src)
        assert len(batches) <= 8 and sum(b['obs']['agent'].shape[0] for b in batches) == 2 * 32
        assert src.steps_per_epoch == 16 and src.global_step == 16
        assert src.state_dict()['collector']['global_step'] == 16

    def test_rollout_needs_value_actor(self):
        src = MultiAgentRolloutSource(match_cfg(), n_steps=2)
        with pytest.raises(TypeError, match='value'):
            src.bind(object())


# ------------------------------------------------------------------ loss / module / model
class TestLossModuleModel:
    def _po(self, n=8, k=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        logits = torch.randn(n, k, generator=g, requires_grad=True)
        return PolicyOutput(torch.distributions.Categorical(logits=logits), torch.randn(n, generator=g))

    def _targets(self, n=8, k=3, seed=1):
        g = torch.Generator().manual_seed(seed)
        return {'actions': torch.randint(k, (n,), generator=g), 'log_prob': torch.full((n,), -1.1),
                'values': torch.randn(n, generator=g), 'advantages': torch.randn(n, generator=g),
                'returns': torch.randn(n, generator=g)}

    def test_single_group_matches_agent_loss(self):
        po, y = self._po(), self._targets()
        single = PPOLoss(ent_coef=0.01)
        multi = MultiAgentLoss(PPOLoss(ent_coef=0.01))
        assert torch.allclose(multi({'g': po}, {'g': y}), single(po, y))
        assert multi.last_metrics['policy_loss/g'] == single.last_metrics['policy_loss']
        assert multi.last_metrics['policy_loss'] == single.last_metrics['policy_loss']

    def test_groups_summed_weighted_and_skipped(self):
        po1, po2, y1, y2 = self._po(seed=0), self._po(seed=2), self._targets(seed=1), self._targets(seed=3)
        a, b = A2CLoss()(po1, y1), A2CLoss()(po2, y2)
        loss = MultiAgentLoss(A2CLoss(), weights={'b': 0.5})
        assert torch.allclose(loss({'a': po1, 'b': po2}, {'a': y1, 'b': y2}), a + 0.5 * b)
        assert set(loss.last_metrics) >= {'entropy/a', 'entropy/b', 'entropy'}
        # a group without targets this minibatch is skipped
        assert torch.allclose(loss({'a': po1, 'b': po2}, {'a': y1}), a)
        with pytest.raises(ValueError, match='no agent group'):
            loss({'a': po1}, {'b': y2})

    def test_kl_aggregates_to_worst_group(self):
        loss = MultiAgentLoss(PPOLoss())
        loss({'a': self._po(seed=0), 'b': self._po(seed=2)}, {'a': self._targets(), 'b': self._targets(seed=5)})
        m = loss.last_metrics
        assert m['approx_kl'] == max(m['approx_kl/a'], m['approx_kl/b'])

    def test_per_group_losses(self):
        loss = MultiAgentLoss({'a': PPOLoss(), 'b': A2CLoss()})
        loss({'a': self._po(), 'b': self._po(seed=2)}, {'a': self._targets(), 'b': self._targets()})
        assert 'clip_frac/a' in loss.last_metrics and 'clip_frac/b' not in loss.last_metrics
        with pytest.raises(KeyError, match="group 'c'"):
            loss({'c': self._po()}, {'c': self._targets()})

    def test_module(self):
        groups = AgentGroups.from_env(MatchEnv())
        module = MultiAgentActorCritic(groups.spaces_by_group(), hidden=(16,))
        assert module.groups == ['agent', 'scout']
        out = module({'agent': torch.zeros(5, 3)})
        assert set(out) == {'agent'} and out['agent'].value.shape == (5,)
        with pytest.raises(KeyError, match='unknown groups'):
            module({'nope': torch.zeros(1, 3)})
        custom = MultiAgentActorCritic(policies={'x': ActorCritic(*groups.spaces_by_group()['scout'])})
        assert custom.groups == ['x']
        with pytest.raises(ValueError, match='exactly one'):
            MultiAgentActorCritic()

    def test_model_act_value_and_registry(self, tmp_path):
        assert _DragonModel._registry['marl-model'] is MultiAgentRLModel
        col = MultiAgentCollector(match_cfg(n_envs=2))
        model = MultiAgentRLModel(MultiAgentActorCritic(col.groups.spaces_by_group()),
                                  MultiAgentLoss(PPOLoss()), model_cfg(tmp_path))
        out = model.act(col.obs)
        assert out.action['agent'].shape == (2, 2) and out.action['scout'].shape == (2, 1)
        assert out.extras['log_prob']['agent'].shape == (2, 2)
        assert out.extras['value']['scout'].shape == (2, 1)
        values = model.value({g: o.astype('float64') for g, o in col.obs.items()})
        assert values['agent'].dtype == torch.float32 and values['agent'].shape == (2, 2)
        greedy = model.act(col.obs, deterministic=True).action
        assert torch.equal(greedy['agent'], model.act(col.obs, deterministic=True).action['agent'])

    def test_evaluate_reports_group_returns(self, tmp_path):
        col = MultiAgentCollector(match_cfg(n_envs=1))
        model = MultiAgentRLModel(MultiAgentActorCritic(col.groups.spaces_by_group()),
                                  MultiAgentLoss(PPOLoss()), model_cfg(tmp_path))
        score = model.evaluate(MultiAgentEpisodeSource(match_cfg(n_envs=2), n_episodes=3))
        assert score == model.last_eval['return']
        assert {'return/agent', 'return/scout', 'length'} <= set(model.last_eval)
        assert model.last_eval['length'] == 8


# ------------------------------------------------------------------ trainers
def ppo_cfg(tmp_path, env=None, total_timesteps=6000, **kw):
    kw.setdefault('rollout', RolloutConfig(n_steps=32, n_minibatches=4, update_epochs=4))
    kw.setdefault('eval', EvalConfig(every_steps=2000, n_episodes=8, n_envs=4))
    kw.setdefault('model', DragonRLConfig(checkpoint_dir=tmp_path / 'ckpt', save_every=1000,
                                          total_timesteps=total_timesteps))
    return MultiAgentPPOTrainerConfig(env=env or match_cfg(n_envs=4), seed=0, **kw)


class TestTrainers:
    def test_config(self, tmp_path):
        cfg = ppo_cfg(tmp_path)
        # a plain DragonRLConfig becomes the algorithm's multi-agent model config
        assert type(cfg.model) is MultiAgentPPOConfig and cfg.model.kind == 'marl-ppo-model'
        assert cfg.model.clip_coef == 0.2 and cfg.rollout.update_epochs == 4  # PPO defaults
        a2c = MultiAgentA2CTrainerConfig(env=match_cfg(), model=DragonRLConfig(checkpoint_dir=tmp_path, save_every=1))
        assert type(a2c.model) is MultiAgentA2CConfig
        assert a2c.rollout.n_steps == 5 and a2c.optim.name == 'rmsprop'
        # loss settings sit on the model config, at the same paths as single-agent PPO
        assert apply_overrides(cfg, {'model.clip_coef': 0.1}).model.clip_coef == 0.1
        assert apply_overrides(cfg, {'env.team_return': 'mean'}).env.team_return == 'mean'
        assert cfg.eval_env_config().seed == cfg.env.seed + cfg.eval.seed_offset

    def test_single_agent_model_config_converts(self, tmp_path):
        single = PPOConfig(checkpoint_dir=tmp_path, save_every=3, clip_coef=0.3, ent_coef=0.01)
        cfg = MultiAgentPPOTrainerConfig(env=match_cfg(), model=single)
        # same algorithm: loss settings carried over; kind becomes the multi-agent one
        assert type(cfg.model) is MultiAgentPPOConfig and cfg.model.kind == 'marl-ppo-model'
        assert (cfg.model.save_every, cfg.model.clip_coef, cfg.model.ent_coef) == (3, 0.3, 0.01)
        # another algorithm: only the shared settings
        other = MultiAgentA2CTrainerConfig(env=match_cfg(), model=single).model
        assert other.save_every == 3 and other.ent_coef == 0.0

    def test_model_classes(self, tmp_path):
        assert _DragonModel._registry['marl-ppo-model'] is MultiAgentPPOModel
        assert _DragonModel._registry['marl-a2c-model'] is MultiAgentA2CModel
        groups = AgentGroups.from_env(MatchEnv())
        module = MultiAgentActorCritic(groups.spaces_by_group())
        model = MultiAgentPPOModel(module, MultiAgentPPOConfig(checkpoint_dir=tmp_path, save_every=1,
                                                               clip_coef=0.15))
        assert isinstance(model.loss_fn, MultiAgentLoss) and model.loss_fn.loss.clip_coef == 0.15
        a2c = MultiAgentA2CModel(module, MultiAgentA2CConfig(checkpoint_dir=tmp_path, save_every=1))
        assert isinstance(a2c.loss_fn.loss, A2CLoss)
        with pytest.raises(TypeError, match='MultiAgentPPOConfig'):
            MultiAgentPPOModel(module, PPOConfig(checkpoint_dir=tmp_path, save_every=1))
        cfg = ppo_cfg(tmp_path, eval=None, model=MultiAgentPPOConfig(
            checkpoint_dir=tmp_path, save_every=1000, total_timesteps=64, ent_coef=0.02))
        with MultiAgentPPOTrainer(cfg) as trainer:
            assert type(trainer.model) is MultiAgentPPOModel
            assert trainer.model.loss_fn.loss.ent_coef == 0.02

    def test_rejects_wrong_config(self, tmp_path):
        cfg = ppo_cfg(tmp_path)
        with pytest.raises(TypeError, match='PPOTrainerConfig'):
            PPOTrainer(cfg)
        with pytest.raises(TypeError, match='MultiAgentA2CTrainerConfig'):
            MultiAgentA2CTrainer(cfg)

    def test_validate_env(self, tmp_path):
        class BadObs(MatchEnv):
            def _observe(self):
                return {a: o * 5 for a, o in super()._observe().items()}
        cfg = ppo_cfg(tmp_path, env=MultiAgentEnvConfig(env_fn=BadObs), validate_env=True)
        with pytest.raises(EnvCheckError, match="env failed"):
            MultiAgentPPOTrainer(cfg)

    def test_ppo_learns(self, tmp_path):
        with MultiAgentPPOTrainer(ppo_cfg(tmp_path)) as trainer:
            before = trainer.evaluate()
            history = trainer.fit()
            after = trainer.evaluate()
        # random play scores ~2 * 8/3 + 1.5 ~ 6.8; perfect play 2 * 8 + 3 = 19
        assert before < 10 and after > 15, (before, after)
        last = history[-1]
        assert last['global_step'] >= 6000
        assert {'val_return', 'val_return_smooth', 'val_return/agent', 'val_return/scout',
                'policy_loss/agent', 'policy_loss/scout', 'approx_kl', 'episode_return/agent'} <= set(last)

    def test_a2c_on_gym_env_runs(self, tmp_path):
        cfg = MultiAgentA2CTrainerConfig(
            env=team_cfg(n_envs=4), seed=0,
            model=DragonRLConfig(checkpoint_dir=tmp_path / 'ckpt', save_every=1000, total_timesteps=400),
            eval=EvalConfig(every_steps=200, n_episodes=4))
        with MultiAgentA2CTrainer(cfg) as trainer:
            history = trainer.fit()
        assert history[-1]['global_step'] >= 400 and math.isfinite(history[-1]['train_loss'])
        assert 'val_return/a' in history[-1]

    def test_resume(self, tmp_path):
        def cfg(total, **kw):
            return ppo_cfg(tmp_path, total_timesteps=total, eval=None, env=match_cfg(n_envs=4),
                           rollout=RolloutConfig(n_steps=32, n_minibatches=2, normalize_obs=True,
                                                 normalize_reward=True), **kw)
        with MultiAgentPPOTrainer(cfg(256)) as first:
            first.fit()
            weights = {k: v.clone() for k, v in first.model.torch_module.state_dict().items()}
        with MultiAgentPPOTrainer(cfg(512, resume=True)) as second:
            second.resume()
            for k, v in second.model.torch_module.state_dict().items():
                assert torch.equal(v, weights[k])
            history = second.fit()  # binding restores the source: steps, normalisers
        assert [r['global_step'] for r in history] == [384, 512]
        assert not np.allclose(second.source.collector.obs_normalizers['agent'].rms.mean, 0)
        assert second.source.collector.obs_normalizers['agent'].rms.count > 256

    def test_custom_module(self, tmp_path):
        groups = AgentGroups.from_env(MatchEnv())
        module = MultiAgentActorCritic(groups.spaces_by_group(), hidden=(8,))
        with MultiAgentPPOTrainer(ppo_cfg(tmp_path, eval=None), module=module) as t:
            assert t.model.torch_module is module
