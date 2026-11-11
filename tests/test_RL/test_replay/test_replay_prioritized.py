import io

import gymnasium as gym
import numpy as np
import pytest
import torch
from dragon_torch.modules.RL.replay.base import _BaseReplayBuffer, validate_buffer
from dragon_torch.modules.RL.replay.prioritized import (PrioritizedReplayBuffer, PrioritizedSample,
                                                        TorchPrioritizedReplayBuffer)

OBS_DIM = (3,)
BUFFERS = [PrioritizedReplayBuffer, TorchPrioritizedReplayBuffer]
DRAWS = 40_000


@pytest.fixture(params=BUFFERS, ids=['numpy', 'torch'])
def cls(request):
    return request.param


def make(cls, maxlength=8, seed=0, **kw):
    return cls(action_dim=1, obs_dim=OBS_DIM, maxlength=maxlength, seed=seed,
               action_dtype=np.int64, **kw)


def step(tags, terminated=False, truncated=False):
    """obs = t, action = t, reward = 10 t, next_obs = t + 1; a scalar tag is one env."""
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
    return PrioritizedSample(*(np_(x) for x in sample))


def tags(x):
    return np_(x)[:, 0].astype(int).tolist()


def frequencies(buf, n, draws=DRAWS):
    return np.bincount(np.asarray(tags(buf.sample(draws).obs)), minlength=n) / draws


def assert_freq(freq, weights, draws=DRAWS):
    probs = np.asarray(weights, np.float64) / np.sum(weights)
    tol = 5 * np.sqrt(probs * (1 - probs) / draws) + 1e-9   # 5 sd of a binomial frequency
    assert np.all(np.abs(freq - probs) <= tol), f'{freq} vs {probs}'


def priorities(buf):
    return np_(buf.priorities)[:buf.size].tolist()


class TestConstruction:
    @pytest.mark.parametrize('kind,cls,backend', [
        ('prioritized-numpy', PrioritizedReplayBuffer, np.ndarray),
        ('prioritized-torch', TorchPrioritizedReplayBuffer, torch.Tensor)])
    def test_kind_and_backend(self, kind, cls, backend):
        assert _BaseReplayBuffer._registry[kind] is cls and cls.backend is backend

    def test_valid_buffer(self, cls):
        buf = make(cls)
        assert validate_buffer(buf) is buf

    def test_defaults(self, cls):
        buf = make(cls)
        assert (buf.alpha, buf.beta, buf.eps, buf.max_priority) == (0.6, 0.4, 1e-6, 1.0)

    def test_priorities_field(self, cls):
        buf = make(cls)
        assert 'priorities' in buf.specs
        assert isinstance(buf.priorities, cls.backend) and tuple(buf.priorities.shape) == (8,)
        assert np_(buf.priorities).dtype == np.float64

    @pytest.mark.parametrize('kw', [{'alpha': -0.1}, {'beta': -1.0}, {'eps': 0.0}])
    def test_invalid_hyperparameters(self, cls, kw):
        with pytest.raises(ValueError):
            make(cls, **kw)

    def test_from_env_forwards_hyperparameters(self, cls):
        buf = cls.from_env(gym.make('CartPole-v1'), maxlength=8, alpha=0.3, beta=0.7)
        assert (buf.alpha, buf.beta, buf.obs_dim) == (0.3, 0.7, (4,))


class TestAdd:
    def test_new_rows_get_max_priority(self, cls):
        buf = make(cls)
        fill(buf, 2)
        assert priorities(buf) == [1.0, 1.0]
        buf.update_priorities([0], [3.0])
        buf.add(**step([2, 3]))   # vector step
        assert priorities(buf) == pytest.approx([3.0, 1.0, 3.0, 3.0])

    def test_evicted_slot_gets_new_priority(self, cls):
        buf = make(cls, maxlength=2, alpha=1.0)
        fill(buf, 2)
        buf.update_priorities([0, 1], [0.0, 1.0])
        fill(buf, 1, start=2)   # overwrites tag 0's slot
        assert tags(buf.all().obs) == [1, 2]
        assert priorities(buf)[0] == pytest.approx(1.0)

    def test_oldest_evicted_first(self, cls):
        buf = make(cls, maxlength=4)
        fill(buf, 10)
        assert len(buf) == 4 and tags(buf.all().obs) == [6, 7, 8, 9]

    def test_bad_shape_rejected(self, cls):
        buf = make(cls)
        with pytest.raises(ValueError):
            buf.add(**{**step([1, 2]), 'obs': np.zeros((2, 4))})
        assert len(buf) == 0


class TestSample:
    def test_empty_raises(self, cls):
        with pytest.raises(ValueError, match='empty'):
            make(cls).sample(1)

    def test_returns_prioritized_sample(self, cls):
        buf = make(cls)
        fill(buf, 5)
        s = buf.sample(6)
        assert isinstance(s, PrioritizedSample)
        assert {k: tuple(v.shape) for k, v in s._asdict().items()} == {
            'obs': (6, 3), 'actions': (6, 1), 'rewards': (6,), 'next_obs': (6, 3),
            'terminated': (6,), 'truncated': (6,), 'idxs': (6,), 'weights': (6,)}
        assert np_(s.weights).dtype == np.float32

    def test_fields_aligned(self, cls):
        buf = make(cls, maxlength=16, alpha=1.0)
        fill(buf, 40)
        buf.update_priorities(np.arange(16), np.arange(16) + 1.0)
        s = as_np(buf.sample(256))
        t = s.obs[:, 0]
        np.testing.assert_array_equal(s.next_obs, s.obs + 1)
        np.testing.assert_array_equal(s.actions[:, 0], t)
        np.testing.assert_array_equal(s.rewards, 10 * t)
        np.testing.assert_array_equal(np_(buf.obs)[s.idxs], s.obs)

    def test_new_rows_sampled_uniformly(self, cls):
        buf = make(cls)
        fill(buf, 4)
        assert_freq(frequencies(buf, 4), np.ones(4))

    @pytest.mark.parametrize('alpha', [1.0, 0.5])
    def test_proportional_to_priority_pow_alpha(self, cls, alpha):
        buf = make(cls, alpha=alpha)
        fill(buf, 4)
        p = np.array([1.0, 2.0, 3.0, 4.0])
        buf.update_priorities(np.arange(4), p)
        assert_freq(frequencies(buf, 4), p ** alpha)

    def test_alpha_zero_is_uniform(self, cls):
        buf = make(cls, alpha=0.0)
        fill(buf, 4)
        buf.update_priorities(np.arange(4), [1.0, 100.0, 0.01, 5.0])
        assert_freq(frequencies(buf, 4), np.ones(4))

    def test_zero_td_error_still_sampled(self, cls):
        buf = make(cls, alpha=1.0, eps=0.1)
        fill(buf, 2)
        buf.update_priorities([0, 1], [0.0, 1.0])
        assert_freq(frequencies(buf, 2), [0.1, 1.1])

    def test_without_replacement(self, cls):
        buf = make(cls)
        fill(buf, 8)
        assert len(set(np_(buf.sample(8, replace=False).idxs).tolist())) == 8
        with pytest.raises(ValueError):
            buf.sample(9, replace=False)

    @pytest.mark.parametrize('n', [3, 8, 11])
    def test_cer_includes_newest(self, cls, n):
        buf = make(cls, maxlength=8)
        fill(buf, n)
        assert as_np(buf.sample(4, use_cer=True)).obs[0, 0] == n - 1

    def test_seeded(self, cls):
        def draw(seed):
            buf = make(cls, seed=seed)
            fill(buf, 8)
            return np_(buf.sample(32).idxs).tolist()
        assert draw(1) == draw(1) and draw(1) != draw(2)

    def test_backends_draw_the_same_indices(self):
        bufs = [make(c, seed=4, alpha=1.0) for c in BUFFERS]
        for buf in bufs:
            fill(buf, 8)
            buf.update_priorities(np.arange(8), np.arange(8) + 1.0)
        a, b = (np_(buf.sample(32).idxs).tolist() for buf in bufs)
        assert a == b


class TestWeights:
    def sample(self, cls, beta):
        buf = make(cls, alpha=1.0, beta=beta)
        fill(buf, 4)
        buf.update_priorities(np.arange(4), [1.0, 2.0, 3.0, 4.0])
        s = as_np(buf.sample(4096))
        return s.obs[:, 0].astype(int), s.weights

    def test_beta_zero_gives_unit_weights(self, cls):
        _, w = self.sample(cls, beta=0.0)
        np.testing.assert_allclose(w, 1.0)

    @pytest.mark.parametrize('beta', [0.4, 1.0])
    def test_importance_sampling_ratio(self, cls, beta):
        # w_i / w_j = (P_j / P_i)^β
        t, w = self.sample(cls, beta)
        P = np.array([1.0, 2.0, 3.0, 4.0]) / 10
        by_tag = {i: w[t == i] for i in range(4)}
        for i, ws in by_tag.items():
            np.testing.assert_allclose(ws, ws[0], rtol=1e-5)   # depends only on the row
            np.testing.assert_allclose(ws[0] / by_tag[0][0], (P[0] / P[i]) ** beta, rtol=1e-4)

    def test_normalised_to_batch_max(self, cls):
        _, w = self.sample(cls, beta=1.0)
        assert w.max() == pytest.approx(1.0) and w.min() > 0

    def test_beta_can_be_annealed(self, cls):
        buf = make(cls, alpha=1.0, beta=0.0)
        fill(buf, 4)
        buf.update_priorities(np.arange(4), [1.0, 2.0, 3.0, 4.0])
        buf.beta = 1.0
        assert np_(buf.sample(256).weights).min() < 1.0


class TestUpdatePriorities:
    def test_sets_abs_td_plus_eps(self, cls):
        buf = make(cls, eps=0.5)
        fill(buf, 3)
        buf.update_priorities([0, 2], [-2.0, 1.0])
        assert priorities(buf) == pytest.approx([2.5, 1.0, 1.5])

    def test_tracks_max_priority(self, cls):
        buf = make(cls, eps=0.0 + 1e-6)
        fill(buf, 3)
        buf.update_priorities([0], [5.0])
        buf.update_priorities([0], [0.1])   # lowering a row doesn't lower the max
        assert buf.max_priority == pytest.approx(5.0)

    def test_accepts_sampled_tensors(self, cls):
        buf = make(cls)
        fill(buf, 4)
        s = buf.sample(4, to_tensor=True)
        buf.update_priorities(s.idxs, torch.full((4,), 2.0, requires_grad=True))
        assert all(priorities(buf)[i] == pytest.approx(2.0) for i in np_(s.idxs))

    def test_empty_update_is_a_no_op(self, cls):
        buf = make(cls)
        fill(buf, 2)
        buf.update_priorities([], [])
        assert priorities(buf) == [1.0, 1.0] and buf.max_priority == 1.0

    def test_length_mismatch(self, cls):
        buf = make(cls)
        fill(buf, 3)
        with pytest.raises(ValueError, match='indices'):
            buf.update_priorities([0, 1], [1.0])

    @pytest.mark.parametrize('bad', [np.nan, np.inf])
    def test_non_finite(self, cls, bad):
        buf = make(cls)
        fill(buf, 3)
        with pytest.raises(ValueError, match='finite'):
            buf.update_priorities([0], [bad])
        buf.sample(4)   # still samples

    @pytest.mark.parametrize('idx', [-1, 3, 8])
    def test_out_of_range(self, cls, idx):
        buf = make(cls)
        fill(buf, 3)
        with pytest.raises(IndexError):
            buf.update_priorities([idx], [1.0])


class TestOutputs:
    def test_numpy_by_default(self):
        buf = make(PrioritizedReplayBuffer)
        fill(buf, 3)
        assert all(isinstance(x, np.ndarray) for x in buf.sample(4))

    def test_to_tensor(self, cls):
        buf = make(cls)
        fill(buf, 3)
        s = buf.sample(4, to_tensor=True)
        assert all(isinstance(x, torch.Tensor) for x in s)
        assert s.weights.dtype == torch.float32 and s.idxs.dtype == torch.long

    def test_torch_returns_tensors(self):
        buf = make(TorchPrioritizedReplayBuffer)
        fill(buf, 3)
        assert all(isinstance(x, torch.Tensor) for x in buf.sample(4))
        assert all(isinstance(x, torch.Tensor) for x in buf.all())

    @pytest.mark.parametrize('to_tensor', [False, True])
    def test_sample_obs(self, cls, to_tensor):
        buf = make(cls)
        fill(buf, 3)
        obs = buf.sample_obs(10, to_tensor=to_tensor, device='cpu')
        assert isinstance(obs, torch.Tensor if to_tensor else cls.backend)
        assert tuple(obs.shape) == (10, 3) and set(tags(obs)) <= {0, 1, 2}

    def test_sample_obs_empty_raises(self, cls):
        with pytest.raises(ValueError):
            make(cls).sample_obs(1)

    def test_all(self, cls):
        buf = make(cls, maxlength=4)
        fill(buf, 6)
        s = as_np(buf.all())
        assert tags(s.obs) == [2, 3, 4, 5] and s.idxs.tolist() == [2, 3, 0, 1]
        np.testing.assert_array_equal(s.weights, np.ones(4, np.float32))

    def test_all_truncate_to(self, cls):
        buf = make(cls)
        fill(buf, 8)
        assert len(set(tags(buf.all(truncate_to=5).obs))) == 5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
    def test_cuda(self):
        buf = make(TorchPrioritizedReplayBuffer, device='cuda')
        fill(buf, 4)
        s = buf.sample(4)
        assert all(x.is_cuda for x in s)
        buf.update_priorities(s.idxs, torch.ones(4, device='cuda'))


class TestState:
    def round_trip(self, state):
        f = io.BytesIO()
        torch.save(state, f)
        f.seek(0)
        return torch.load(f, weights_only=True)

    def filled(self, cls):
        buf = make(cls, alpha=1.0, beta=0.4)
        fill(buf, 4)
        buf.update_priorities(np.arange(4), [1.0, 2.0, 7.0, 4.0])
        buf.beta = 0.9   # annealed
        return buf

    def test_round_trip(self, cls):
        src = self.filled(cls)
        dst = make(cls, alpha=1.0)
        dst.load_state_dict(self.round_trip(src.state_dict()))
        assert priorities(dst) == pytest.approx(priorities(src))
        assert dst.max_priority == pytest.approx(7.0) and dst.beta == 0.9
        assert_freq(frequencies(dst, 4), [1.0, 2.0, 7.0, 4.0])

    def test_new_rows_after_load_get_restored_max(self, cls):
        dst = make(cls, alpha=1.0)
        dst.load_state_dict(self.filled(cls).state_dict())
        fill(dst, 1, start=4)
        assert priorities(dst)[-1] == pytest.approx(7.0)

    def test_rng_resumes(self, cls):
        src = self.filled(cls)
        dst = make(cls, alpha=1.0, seed=99)
        dst.load_state_dict(src.state_dict())
        assert np_(dst.sample(16).idxs).tolist() == np_(src.sample(16).idxs).tolist()

    @pytest.mark.parametrize('src_cls,dst_cls', [(PrioritizedReplayBuffer, TorchPrioritizedReplayBuffer),
                                                 (TorchPrioritizedReplayBuffer, PrioritizedReplayBuffer)])
    def test_across_backends(self, src_cls, dst_cls):
        src = self.filled(src_cls)
        dst = make(dst_cls, alpha=1.0)
        dst.load_state_dict(self.round_trip(src.state_dict()))
        assert isinstance(dst.priorities, dst_cls.backend)
        assert priorities(dst) == pytest.approx(priorities(src))

    def test_uniform_state_rejected(self, cls):
        # a uniform buffer has no priorities field, so the layouts differ
        from dragon_torch.modules.RL.replay.uniform import UniformReplayBuffer
        src = make(UniformReplayBuffer)
        fill(src, 2)
        with pytest.raises((ValueError, KeyError)):
            make(cls).load_state_dict(src.state_dict())


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
        s = buf.sample(16, to_tensor=True)
        buf.update_priorities(s.idxs, torch.randn(16))
        assert len(buf) == 50 and tuple(s.obs.shape) == (16, 4)
        env.close()
