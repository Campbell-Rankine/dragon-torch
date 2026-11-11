import io

import gymnasium as gym
import numpy as np
import pytest
import torch
from dragon_torch.data.envs import DragonEnvConfig
from dragon_torch.data.RL import ActOutput, EnvCollector
from dragon_torch.modules.RL.replay.base import ReplaySample, _BaseReplayBuffer, validate_buffer
from dragon_torch.modules.RL.replay.uniform import TorchUniformReplayBuffer, UniformReplayBuffer

OBS_DIM = (3,)
BUFFERS = [UniformReplayBuffer, TorchUniformReplayBuffer]


@pytest.fixture(params=BUFFERS, ids=['numpy', 'torch'])
def cls(request):
    return request.param


def make(cls, maxlength=8, seed=0, **kw):
    return cls(action_dim=1, obs_dim=OBS_DIM, maxlength=maxlength, seed=seed,
               action_dtype=np.int64, **kw)


def step(tags, terminated=False, truncated=False):
    """A vector step whose fields all encode its tag: obs = t, action = t, reward = 10 t,
    next_obs = t + 1. A scalar tag gives a single-env step."""
    t = np.asarray(tags, np.float32)
    obs = np.broadcast_to(t[..., None], (*t.shape, *OBS_DIM)).copy()
    return dict(obs=obs, action=t.astype(np.int64), reward=10 * t, next_obs=obs + 1,
                terminated=np.full(t.shape, terminated), truncated=np.full(t.shape, truncated))


def fill(buf, n, start=0):
    for t in range(start, start + n):
        buf.add(**step(t))


def np_(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def as_np(sample):
    return ReplaySample(*(np_(x) for x in sample))


def tags(x):
    return np_(x)[:, 0].astype(int).tolist()


class TestRegistration:
    @pytest.mark.parametrize('kind,cls,backend', [('uniform-numpy', UniformReplayBuffer, np.ndarray),
                                                  ('torch-uniform', TorchUniformReplayBuffer, torch.Tensor)])
    def test_kind_and_backend(self, kind, cls, backend):
        assert _BaseReplayBuffer._registry[kind] is cls and cls.backend is backend

    def test_valid_buffer(self, cls):
        buf = make(cls)
        assert validate_buffer(buf) is buf

    def test_storage_backend(self, cls):
        buf = make(cls)
        assert all(isinstance(getattr(buf, name), cls.backend) for name in buf.specs)


class TestAdd:
    def test_single_env(self, cls):
        buf = make(cls)
        buf.add(**step(5))
        assert len(buf) == 1 and tags(buf.all().obs) == [5]

    def test_vector_env(self, cls):
        buf = make(cls)
        buf.add(**step([1, 2, 3]))
        assert len(buf) == 3 and tags(buf.all().obs) == [1, 2, 3]

    def test_len_capped(self, cls):
        buf = make(cls, maxlength=4)
        fill(buf, 10)
        assert len(buf) == 4

    def test_oldest_evicted_first(self, cls):
        buf = make(cls, maxlength=4)
        fill(buf, 10)
        assert tags(buf.all().obs) == [6, 7, 8, 9]

    def test_vector_step_wraps(self, cls):
        buf = make(cls, maxlength=4)
        buf.add(**step([0, 1, 2]))
        buf.add(**step([3, 4, 5]))
        assert tags(buf.all().obs) == [2, 3, 4, 5]

    def test_copies_input(self, cls):
        buf = make(cls)
        tr = step([1])
        buf.add(**tr)
        tr['obs'][:] = 99
        assert tags(buf.all().obs) == [1]

    def test_torch_input(self, cls):
        buf = make(cls)
        buf.add(**{k: torch.as_tensor(v) for k, v in step([1, 2]).items()})
        assert tags(buf.all().obs) == [1, 2]

    def test_bad_shape_leaves_buffer_untouched(self, cls):
        buf = make(cls)
        with pytest.raises(ValueError):
            buf.add(**{**step([1, 2]), 'obs': np.zeros((2, 4))})
        assert len(buf) == 0 and buf.ptr == 0


class TestSample:
    def test_empty_raises(self, cls):
        with pytest.raises(ValueError, match='empty'):
            make(cls).sample(1)

    def test_returns_replay_sample(self, cls):
        buf = make(cls)
        fill(buf, 5)
        s = buf.sample(6)
        assert isinstance(s, ReplaySample)
        assert {k: tuple(v.shape) for k, v in s._asdict().items()} == {
            'obs': (6, 3), 'actions': (6, 1), 'rewards': (6,), 'next_obs': (6, 3),
            'terminated': (6,), 'truncated': (6,), 'idxs': (6,)}

    def test_dtypes(self, cls):
        buf = make(cls)
        fill(buf, 3)
        s = as_np(buf.sample(4))
        assert s.obs.dtype == s.next_obs.dtype == s.rewards.dtype == np.float32
        assert s.actions.dtype == np.int64 and s.terminated.dtype == s.truncated.dtype == np.bool_

    def test_fields_aligned(self, cls):
        buf = make(cls, maxlength=16)
        fill(buf, 40)
        s = as_np(buf.sample(256))
        t = s.obs[:, 0]
        np.testing.assert_array_equal(s.next_obs, s.obs + 1)
        np.testing.assert_array_equal(s.actions[:, 0], t)
        np.testing.assert_array_equal(s.rewards, 10 * t)
        np.testing.assert_array_equal(np_(buf.obs)[s.idxs], s.obs)  # idxs point at the rows

    def test_only_filled_slots(self, cls):
        buf = make(cls, maxlength=8)
        fill(buf, 3)
        assert set(tags(buf.sample(500).obs)) == {0, 1, 2}

    def test_terminated_and_truncated_kept_apart(self, cls):
        buf = make(cls)
        buf.add(**step(0, terminated=True))
        buf.add(**step(1, truncated=True))
        buf.add(**step(2))
        s = as_np(buf.sample(300))
        t = s.obs[:, 0]
        np.testing.assert_array_equal(s.terminated, t == 0)
        np.testing.assert_array_equal(s.truncated, t == 1)

    def test_without_replacement(self, cls):
        buf = make(cls)
        fill(buf, 8)
        assert len(set(np_(buf.sample(8, replace=False).idxs).tolist())) == 8
        with pytest.raises(ValueError):
            buf.sample(9, replace=False)

    def test_uniform(self, cls):
        buf = make(cls, maxlength=8)
        fill(buf, 8)
        n = 16_000
        counts = np.bincount(np_(buf.sample(n).actions)[:, 0], minlength=8)
        # each count ~ Binomial(n, 1/8); 5 sd is a loose bound
        np.testing.assert_allclose(counts, n / 8, atol=5 * np.sqrt(n / 8 * 7 / 8))

    def test_seeded(self, cls):
        def draw(seed):
            buf = make(cls, seed=seed)
            fill(buf, 8)
            return np_(buf.sample(32).idxs).tolist()
        assert draw(1) == draw(1) and draw(1) != draw(2)

    def test_backends_draw_the_same_indices(self):
        # one seeded numpy RNG drives both, so they're interchangeable in experiments
        bufs = [make(c, seed=4) for c in BUFFERS]
        for buf in bufs:
            fill(buf, 8)
        a, b = (np_(buf.sample(32).idxs).tolist() for buf in bufs)
        assert a == b

    @pytest.mark.parametrize('n', [3, 8, 11])  # before, at and after wrapping
    def test_cer_includes_newest(self, cls, n):
        buf = make(cls, maxlength=8)
        fill(buf, n)
        s = as_np(buf.sample(4, use_cer=True))
        assert s.obs[0, 0] == n - 1 and 0 <= s.idxs[0] < 8

    def test_to_tensor(self, cls):
        buf = make(cls)
        fill(buf, 4)
        s = buf.sample(5, to_tensor=True)
        assert isinstance(s, ReplaySample)
        assert all(isinstance(x, torch.Tensor) and x.device.type == 'cpu' for x in s)
        assert s.obs.dtype == torch.float32 and s.actions.dtype == torch.int64
        assert s.terminated.dtype == torch.bool

    def test_to_tensor_device(self, cls):
        buf = make(cls)
        fill(buf, 4)
        assert buf.sample(2, to_tensor=True, device='cpu').obs.device == torch.device('cpu')


class TestSampleObs:
    def test_shape_and_contents(self, cls):
        buf = make(cls)
        fill(buf, 3)
        obs = buf.sample_obs(10)
        assert isinstance(obs, cls.backend) and tuple(obs.shape) == (10, 3)
        assert set(tags(obs)) <= {0, 1, 2}

    def test_tensor(self, cls):
        buf = make(cls)
        fill(buf, 3)
        assert isinstance(buf.sample_obs(4, to_tensor=True), torch.Tensor)

    def test_empty_raises(self, cls):
        with pytest.raises(ValueError):
            make(cls).sample_obs(1)


class TestAll:
    def test_oldest_first(self, cls):
        buf = make(cls, maxlength=4)
        fill(buf, 6)
        s = as_np(buf.all())
        assert tags(s.obs) == [2, 3, 4, 5] and s.idxs.tolist() == [2, 3, 0, 1]
        np.testing.assert_array_equal(s.rewards, 10 * s.obs[:, 0])

    def test_empty(self, cls):
        assert tuple(make(cls).all().obs.shape) == (0, 3)

    def test_truncate_to(self, cls):
        buf = make(cls, maxlength=8)
        fill(buf, 8)
        s = buf.all(truncate_to=5)
        assert len(s.obs) == 5 and len(set(tags(s.obs))) == 5

    def test_truncate_to_more_than_stored(self, cls):
        buf = make(cls, maxlength=8)
        fill(buf, 3)
        assert sorted(tags(buf.all(truncate_to=10).obs)) == [0, 1, 2]


class TestNumpyOutputs:
    def test_numpy_by_default(self):
        buf = make(UniformReplayBuffer)
        fill(buf, 3)
        assert all(isinstance(x, np.ndarray) for x in buf.sample(4))
        assert all(isinstance(x, np.ndarray) for x in buf.all())


class TestTorchOutputs:
    def test_tensors_without_to_tensor(self):
        buf = make(TorchUniformReplayBuffer)
        fill(buf, 3)
        s = buf.sample(4)
        assert all(isinstance(x, torch.Tensor) for x in s)
        assert s.idxs.dtype == torch.long
        assert all(isinstance(x, torch.Tensor) for x in buf.all())

    def test_sample_is_a_copy(self):
        # advanced indexing copies, so a training step can't corrupt the buffer
        buf = make(TorchUniformReplayBuffer)
        fill(buf, 3)
        s = buf.sample(4)
        s.obs.fill_(99)
        assert set(tags(buf.all().obs)) == {0, 1, 2}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
    def test_cuda(self):
        buf = make(TorchUniformReplayBuffer, device='cuda')
        fill(buf, 3)
        s = buf.sample(4)
        assert all(x.is_cuda for x in s)
        assert buf.sample(4, device='cpu').obs.device.type == 'cpu'


class TestState:
    def round_trip(self, state):
        f = io.BytesIO()
        torch.save(state, f)
        f.seek(0)
        return torch.load(f, weights_only=True)

    def test_round_trip(self, cls):
        src = make(cls, maxlength=4)
        fill(src, 6)
        dst = make(cls, maxlength=4, seed=99)
        dst.load_state_dict(self.round_trip(src.state_dict()))
        assert len(dst) == 4 and dst.ptr == src.ptr
        assert tags(dst.all().obs) == [2, 3, 4, 5]
        np.testing.assert_array_equal(as_np(dst.all()).rewards, as_np(src.all()).rewards)

    def test_eviction_resumes(self, cls):
        src = make(cls, maxlength=4)
        fill(src, 6)
        dst = make(cls, maxlength=4)
        dst.load_state_dict(src.state_dict())
        fill(dst, 2, start=6)
        assert tags(dst.all().obs) == [4, 5, 6, 7]

    def test_rng_resumes(self, cls):
        src = make(cls, seed=5)
        fill(src, 8)
        src.sample(10)
        dst = make(cls, seed=0)
        dst.load_state_dict(src.state_dict())
        assert np_(dst.sample(16).idxs).tolist() == np_(src.sample(16).idxs).tolist()

    def test_state_is_a_snapshot(self, cls):
        buf = make(cls, maxlength=4)
        fill(buf, 2)
        state = buf.state_dict()
        fill(buf, 4, start=2)
        restored = make(cls, maxlength=4)
        restored.load_state_dict(state)
        assert tags(restored.all().obs) == [0, 1]

    def test_load_clears_old_contents(self, cls):
        src = make(cls)
        fill(src, 2)
        dst = make(cls)
        fill(dst, 8, start=100)
        dst.load_state_dict(src.state_dict())
        assert len(dst) == 2 and set(tags(dst.sample(200).obs)) == {0, 1}

    @pytest.mark.parametrize('src_cls,dst_cls', [(UniformReplayBuffer, TorchUniformReplayBuffer),
                                                 (TorchUniformReplayBuffer, UniformReplayBuffer)])
    def test_across_backends(self, src_cls, dst_cls):
        src = make(src_cls, maxlength=4)
        fill(src, 6)
        dst = make(dst_cls, maxlength=4)
        dst.load_state_dict(self.round_trip(src.state_dict()))
        assert isinstance(dst.obs, dst_cls.backend) and tags(dst.all().obs) == [2, 3, 4, 5]


class TestEnvs:
    def test_gymnasium_env_loop(self, cls):
        env = gym.make('CartPole-v1')
        buf = cls.from_env(env, maxlength=50, seed=0)
        obs, _ = env.reset(seed=0)
        for _ in range(80):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            buf.add(obs, action, reward, next_obs, terminated, truncated)
            obs = env.reset()[0] if terminated or truncated else next_obs
        s = as_np(buf.sample(16))
        assert len(buf) == 50 and s.obs.shape == (16, 4) and s.actions.shape == (16, 1)
        assert set(np.unique(s.actions)) <= {0, 1}
        env.close()

    def test_collector_keeps_final_obs(self, cls):
        # 2 envs truncated every 3 steps: the stored next_obs must be the true final obs,
        # not the reset obs the collector hands to the policy
        cfg = DragonEnvConfig(env_fn=lambda: gym.make('Pendulum-v1', max_episode_steps=3),
                              n_envs=2)
        col = EnvCollector(cfg)
        buf = cls.from_env(col, maxlength=64)

        class Actor:
            def act(self, obs, deterministic=False):
                return ActOutput(torch.zeros(obs.shape[0], 1), {})

        transitions = [col.step(Actor()) for _ in range(6)]
        for tr in transitions:
            buf.add_transition(tr)
        s = as_np(buf.all())
        assert len(buf) == 12 and s.actions.shape == (12, 1)
        assert s.truncated.sum() == 4 and not s.terminated.any()
        expected_next = np.concatenate([tr.next_obs for tr in transitions])
        np.testing.assert_allclose(s.next_obs, expected_next, rtol=1e-6)
        col.close()
