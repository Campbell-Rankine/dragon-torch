import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from pydantic import BaseModel, ValidationError
from dragon_torch.data.envs import DragonEnvConfig
from dragon_torch.data.RL import EnvCollector, Transition
from dragon_torch.modules.RL.exceptions import InvalidBackendException
from dragon_torch.modules.RL.replay.base import (BaseReplayBuffer, ReplaySample, _BaseReplayBuffer,
                                                 _ReplayBuffer, validate_buffer)

BACKENDS = [np.ndarray, torch.Tensor]
FIELDS = ['obs', 'next_obs', 'actions', 'rewards', 'terminated', 'truncated']


class Stub:
    """No-op versions of _ReplayBuffer's abstract methods."""
    def add(self, *a, **k): ...
    def sample(self, *a, **k): ...
    def sample_obs(self, *a, **k): ...
    def all(self, *a, **k): ...


@pytest.fixture(autouse=True)
def registry():
    """Undo the registrations made by the throwaway subclasses below."""
    saved = dict(_BaseReplayBuffer._registry)
    yield _BaseReplayBuffer._registry
    _BaseReplayBuffer._registry.clear()
    _BaseReplayBuffer._registry.update(saved)


def buffer_cls(backend, kind='test'):
    class Buffer(Stub, _ReplayBuffer[backend]):
        pass
    Buffer.kind = kind
    _BaseReplayBuffer._registry[kind] = Buffer
    return Buffer


@pytest.fixture(params=BACKENDS, ids=['numpy', 'torch'])
def cls(request):
    return buffer_cls(request.param)


def make(cls, **kw):
    return cls(**{'action_dim': 2, 'obs_dim': 4, 'maxlength': 5, **kw})


class TestRegistry:
    def test_declared_kind(self, registry):
        class Uniform(Stub, _ReplayBuffer[np.ndarray]):
            kind = 'test-uniform'
        assert registry['test-uniform'] is Uniform

    def test_undeclared_kind_uses_class_name(self, registry):
        class Unnamed(Stub, _ReplayBuffer[np.ndarray]):
            pass
        assert registry['Unnamed'] is Unnamed

    def test_inherited_kind_not_overwritten(self, registry):
        class Parent(Stub, _ReplayBuffer[np.ndarray]):
            kind = 'test-parent'

        class Child(Parent):
            pass
        assert registry['test-parent'] is Parent and registry['Child'] is Child

    def test_registry_is_shared(self):
        assert buffer_cls(np.ndarray)._registry is _BaseReplayBuffer._registry


class TestBackendInference:
    @pytest.mark.parametrize('backend', BACKENDS)
    def test_from_generic_parameter(self, backend):
        assert buffer_cls(backend).backend is backend

    def test_explicit_backend_wins(self):
        class Explicit(Stub, _ReplayBuffer[np.ndarray]):
            backend = torch.Tensor
        assert Explicit.backend is torch.Tensor

    def test_inherited_by_subclass(self):
        class Child(buffer_cls(torch.Tensor)):
            pass
        assert Child.backend is torch.Tensor

    def test_unparameterised_has_no_backend(self):
        class Bare(Stub, _ReplayBuffer):
            pass
        assert Bare.backend is None

    def test_base_classes_have_no_backend(self):
        assert _BaseReplayBuffer.backend is None and _ReplayBuffer.backend is None


class TestValidateBuffer:
    def test_returns_buffer(self, cls):
        buf = make(cls)
        assert validate_buffer(buf) is buf

    def test_unsupported_backend(self):
        buf = object.__new__(buffer_cls(list))  # skip __init__, which rejects it too
        with pytest.raises(InvalidBackendException) as err:
            validate_buffer(buf)
        assert err.value.backend_type is list

    def test_unregistered_kind(self, cls, registry):
        buf = make(cls)
        del registry['test']
        with pytest.raises(TypeError, match='test'):
            validate_buffer(buf)


class TestPydanticField:
    class Config(BaseModel):
        buffer: BaseReplayBuffer

    def test_buffer_passes_through(self, cls):
        buf = make(cls)
        assert self.Config(buffer=buf).buffer is buf

    def test_non_buffer_rejected(self):
        with pytest.raises(ValidationError):
            self.Config(buffer=object())

    def test_unsupported_backend_rejected(self):
        # pydantic only wraps ValueError, so this escapes as itself
        with pytest.raises(InvalidBackendException):
            self.Config(buffer=object.__new__(buffer_cls(list)))


class TestInit:
    def test_abstract(self):
        with pytest.raises(TypeError):
            _ReplayBuffer(action_dim=1, obs_dim=1, maxlength=1)

    @pytest.mark.parametrize('backend', [None, list, torch.nn.Module])
    def test_unsupported_backend(self, backend):
        class Bad(Stub, _ReplayBuffer):
            pass
        Bad.backend = backend
        with pytest.raises(InvalidBackendException):
            make(Bad)

    @pytest.mark.parametrize('maxlength', [0, -1])
    def test_maxlength_must_be_positive(self, cls, maxlength):
        with pytest.raises(ValueError):
            make(cls, maxlength=maxlength)

    def test_starts_empty(self, cls):
        buf = make(cls)
        assert buf.ptr == 0 and buf.size == 0 and len(buf) == 0

    @pytest.mark.parametrize('obs_dim,expected', [
        (4, (4,)), (np.int64(4), (4,)), ((4,), (4,)), ((3, 8, 8), (3, 8, 8)), ([2, 3], (2, 3)),
    ])
    def test_obs_dim_normalised(self, cls, obs_dim, expected):
        buf = make(cls, obs_dim=obs_dim)
        assert buf.obs_dim == expected and all(type(d) is int for d in buf.obs_dim)

    def test_attributes(self, cls):
        buf = make(cls, reward_dim=3, action_dtype=np.int64, obs_dtype=np.uint8)
        assert (buf.maxlength, buf.action_dim, buf.reward_dim) == (5, 2, 3)
        assert buf.action_dtype == np.int64 and buf.obs_dtype == np.uint8
        assert buf.device == torch.device('cpu')


class TestAllocation:
    def test_default_fields(self, cls):
        assert list(make(cls).specs) == FIELDS

    def test_backend_type(self, cls):
        buf = make(cls)
        for name in FIELDS:
            assert isinstance(getattr(buf, name), cls.backend), name

    def test_shapes(self, cls):
        buf = make(cls, obs_dim=(3, 8), action_dim=2, maxlength=5)
        shapes = {name: tuple(getattr(buf, name).shape) for name in FIELDS}
        assert shapes == {'obs': (5, 3, 8), 'next_obs': (5, 3, 8), 'actions': (5, 2),
                          'rewards': (5,), 'terminated': (5,), 'truncated': (5,)}

    def test_scalar_rewards_match_flags(self, cls):
        # (N, 1) rewards would broadcast r + γ(1 − d)Q' to (B, B)
        buf = make(cls)
        assert buf.rewards.shape == buf.terminated.shape

    def test_vector_rewards(self, cls):
        assert tuple(make(cls, reward_dim=3).rewards.shape) == (5, 3)

    def test_default_dtypes(self, cls):
        buf = make(cls)
        dtypes = {name: np.dtype(buf.to_np(getattr(buf, name)).dtype) for name in FIELDS}
        assert dtypes == {'obs': np.float32, 'next_obs': np.float32, 'actions': np.float32,
                          'rewards': np.float32, 'terminated': np.bool_, 'truncated': np.bool_}

    def test_custom_dtypes(self, cls):
        buf = make(cls, action_dtype=np.int64, obs_dtype=np.uint8)
        assert buf.to_np(buf.actions).dtype == np.int64
        assert buf.to_np(buf.obs).dtype == buf.to_np(buf.next_obs).dtype == np.uint8

    def test_zeroed(self, cls):
        buf = make(cls)
        for name in FIELDS:
            assert not buf.to_np(getattr(buf, name)).any(), name

    def test_fields_do_not_share_memory(self, cls):
        buf = make(cls)
        buf.obs[0] = 1
        assert not buf.to_np(buf.next_obs).any()

    def test_specs_extendable(self, cls):
        class Prioritized(cls):
            def _buffer_specs(self):
                return {**super()._buffer_specs(), 'priorities': ((), np.dtype(np.float64))}
        buf = make(Prioritized)
        assert list(buf.specs) == FIELDS + ['priorities']
        assert isinstance(buf.priorities, cls.backend) and tuple(buf.priorities.shape) == (5,)
        assert buf.to_np(buf.priorities).dtype == np.float64

    def test_torch_device(self):
        buf = make(buffer_cls(torch.Tensor), device=torch.device('cpu'))
        assert all(getattr(buf, name).device == torch.device('cpu') for name in FIELDS)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
    def test_cuda_device(self):
        buf = make(buffer_cls(torch.Tensor), device='cuda')
        assert all(getattr(buf, name).is_cuda for name in FIELDS)

    def test_numpy_ignores_device(self):
        assert isinstance(make(buffer_cls(np.ndarray), device='cpu').obs, np.ndarray)


@pytest.mark.parametrize('np_dtype,torch_dtype', [
    (np.float32, torch.float32), (np.float64, torch.float64), (np.float16, torch.float16),
    (np.int64, torch.int64), (np.int32, torch.int32), (np.uint8, torch.uint8),
    (np.bool_, torch.bool),
])
def test_torch_dtype(np_dtype, torch_dtype):
    assert _ReplayBuffer.torch_dtype(np.dtype(np_dtype)) == torch_dtype


class TestValidateObs:
    @pytest.mark.parametrize('to', [np.zeros, torch.zeros], ids=['numpy', 'torch'])
    def test_valid_batch(self, cls, to):
        make(cls, obs_dim=(3, 8)).validate_obs(to((6, 3, 8)))

    def test_single_env_batch(self, cls):
        make(cls).validate_obs(np.zeros((1, 4)))

    @pytest.mark.parametrize('shape', [(6, 5), (6, 4, 1), (4,), (6,)])
    def test_wrong_shape(self, cls, shape):
        with pytest.raises(ValueError, match='does not match'):
            make(cls).validate_obs(np.zeros(shape))

    def test_not_an_array(self, cls):
        with pytest.raises(TypeError):
            make(cls).validate_obs([[0.0] * 4])


class TestToNp:
    def test_tensor(self, cls):
        out = make(cls).to_np(torch.ones(3, requires_grad=True))
        assert isinstance(out, np.ndarray) and out.tolist() == [1, 1, 1]

    def test_array_passes_through(self, cls):
        x = np.ones(3)
        assert make(cls).to_np(x) is x

    def test_list(self, cls):
        assert make(cls).to_np([1, 2]).tolist() == [1, 2]


class TestFromSpaces:
    @pytest.mark.parametrize('space,shape,dtype', [
        (spaces.Box(-1, 1, (4,), np.float64), (4,), np.float32),
        (spaces.Box(0, 255, (3, 8, 8), np.uint8), (3, 8, 8), np.uint8),
        (spaces.Box(-1, 1, (), np.float32), (1,), np.float32),
        (spaces.Discrete(5), (1,), np.int64),
        (spaces.MultiDiscrete([2, 3]), (2,), np.int64),
        (spaces.MultiBinary(3), (3,), np.int8),
    ])
    def test_observation_layout(self, cls, space, shape, dtype):
        buf = cls.from_spaces(space, spaces.Discrete(2), maxlength=5)
        assert buf.obs_dim == shape and buf.obs_dtype == dtype

    @pytest.mark.parametrize('space,dim,dtype', [
        (spaces.Box(-1, 1, (3,), np.float64), 3, np.float32),
        (spaces.Discrete(4, start=1), 1, np.int64),
        (spaces.MultiDiscrete([2, 3, 4]), 3, np.int64),
        (spaces.MultiBinary(2), 2, np.int8),
    ])
    def test_action_layout(self, cls, space, dim, dtype):
        buf = cls.from_spaces(spaces.Box(-1, 1, (4,)), space, maxlength=5)
        assert buf.action_dim == dim and buf.action_dtype == dtype
        assert tuple(buf.actions.shape) == (5, dim)

    def test_kwargs_forwarded_and_override(self, cls):
        buf = cls.from_spaces(spaces.Box(-1, 1, (4,)), spaces.Discrete(2), maxlength=5,
                              obs_dtype=np.float16, seed=3, reward_dim=2)
        assert buf.obs_dtype == np.float16 and buf.reward_dim == 2

    @pytest.mark.parametrize('space', [
        spaces.Dict({'a': spaces.Box(-1, 1, (2,))}), spaces.Tuple([spaces.Discrete(2)]),
    ])
    def test_unsupported_space(self, cls, space):
        with pytest.raises(NotImplementedError):
            cls.from_spaces(space, spaces.Discrete(2), maxlength=5)
        with pytest.raises(NotImplementedError):
            cls.from_spaces(spaces.Box(-1, 1, (2,)), space, maxlength=5)

    def test_action_must_be_flat(self, cls):
        with pytest.raises(ValueError, match='1-D'):
            cls.from_spaces(spaces.Box(-1, 1, (4,)), spaces.Box(-1, 1, (2, 2)), maxlength=5)


class TestFromEnv:
    def test_env(self, cls):
        buf = cls.from_env(gym.make('CartPole-v1'), maxlength=5)
        assert (buf.obs_dim, buf.action_dim, buf.action_dtype) == ((4,), 1, np.int64)

    def test_vector_env_uses_single_spaces(self, cls):
        env = gym.make_vec('Pendulum-v1', num_envs=3, vectorization_mode='sync')
        buf = cls.from_env(env, maxlength=5)
        assert (buf.obs_dim, buf.action_dim) == ((3,), 1)
        env.close()

    def test_collector(self, cls):
        col = EnvCollector(DragonEnvConfig(env_fn=lambda: gym.make('CartPole-v1'), n_envs=2))
        buf = cls.from_env(col, maxlength=5)
        assert (buf.obs_dim, buf.action_dim) == ((4,), 1)
        col.close()


class TestBatch:
    def fields(self, n=None, obs_dim=(4,), action=None):
        lead = () if n is None else (n,)
        return dict(obs=np.ones((*lead, *obs_dim)), next_obs=np.ones((*lead, *obs_dim)),
                    actions=np.ones((*lead, 2)) if action is None else action,
                    rewards=np.ones(lead), terminated=np.zeros(lead, bool),
                    truncated=np.zeros(lead, bool))

    def test_single_env_gets_batch_dim(self, cls):
        out = make(cls)._batch(**self.fields())
        assert {k: v.shape for k, v in out.items()} == {
            'obs': (1, 4), 'next_obs': (1, 4), 'actions': (1, 2), 'rewards': (1,),
            'terminated': (1,), 'truncated': (1,)}

    def test_vector_env(self, cls):
        out = make(cls)._batch(**self.fields(n=3))
        assert out['obs'].shape == (3, 4) and out['rewards'].shape == (3,)

    @pytest.mark.parametrize('n,action', [(None, np.int64(1)), (None, np.array([1])),
                                          (3, np.array([0, 1, 1])), (3, np.ones((3, 1)))],
                             ids=['scalar', 'single', 'flat-batch', 'batch'])
    def test_discrete_actions(self, cls, n, action):
        out = make(cls, action_dim=1, action_dtype=np.int64)._batch(**self.fields(n, action=action))
        assert out['actions'].shape == ((n or 1), 1) and out['actions'].dtype == np.int64

    def test_discrete_observations(self, cls):
        buf = cls.from_spaces(spaces.Discrete(5), spaces.Discrete(2), maxlength=5)
        out = buf._batch(**self.fields(n=3, obs_dim=(), action=np.zeros(3)))
        assert out['obs'].shape == (3, 1)

    def test_cast_to_storage_dtype(self, cls):
        buf = make(cls, action_dtype=np.int64)
        out = buf._batch(**self.fields(n=2))
        assert out['obs'].dtype == np.float32 and out['actions'].dtype == np.int64
        assert out['terminated'].dtype == np.bool_

    def test_torch_input(self, cls):
        f = {k: torch.as_tensor(v) for k, v in self.fields(n=2).items()}
        f['actions'] = f['actions'].float().requires_grad_()
        out = make(cls)._batch(**f)
        assert all(isinstance(v, np.ndarray) for v in out.values())

    @pytest.mark.parametrize('field,bad', [('obs', np.ones((2, 5))), ('actions', np.ones((2, 3))),
                                           ('rewards', np.ones(3)), ('truncated', np.zeros(1, bool))])
    def test_shape_mismatch(self, cls, field, bad):
        with pytest.raises(ValueError, match=field):
            make(cls)._batch(**{**self.fields(n=2), field: bad})

    def test_needs_terminated(self, cls):
        f = self.fields(n=2)
        del f['terminated']
        with pytest.raises(ValueError, match='terminated'):
            make(cls)._batch(**f)


class TestStore:
    def batch(self, tags):
        tags = np.asarray(tags, np.float32)
        return {'obs': np.repeat(tags[:, None], 4, 1), 'rewards': tags}

    def stored(self, buf):
        return buf.to_np(buf.rewards)[buf._ordered_idxs()].tolist()

    def test_writes_at_pointer(self, cls):
        buf = make(cls)
        np.testing.assert_array_equal(buf._store(self.batch([1, 2])), [0, 1])
        np.testing.assert_array_equal(buf._store(self.batch([3])), [2])
        assert (buf.ptr, buf.size) == (3, 3) and self.stored(buf) == [1, 2, 3]
        assert buf.to_np(buf.obs)[2].tolist() == [3] * 4

    def test_wraps_and_evicts_oldest(self, cls):
        buf = make(cls, maxlength=4)
        buf._store(self.batch([1, 2, 3]))
        np.testing.assert_array_equal(buf._store(self.batch([4, 5, 6])), [3, 0, 1])
        assert (buf.ptr, buf.size) == (2, 4) and self.stored(buf) == [3, 4, 5, 6]

    def test_batch_larger_than_buffer_keeps_newest(self, cls):
        buf = make(cls, maxlength=3)
        buf._store(self.batch([1]))
        buf._store(self.batch([2, 3, 4, 5, 6]))
        assert buf.size == 3 and self.stored(buf) == [4, 5, 6]

    def test_ordered_idxs(self, cls):
        buf = make(cls, maxlength=4)
        assert buf._ordered_idxs().tolist() == []
        buf._store(self.batch([1, 2]))
        assert buf._ordered_idxs().tolist() == [0, 1]
        buf._store(self.batch([3, 4, 5]))
        assert buf._ordered_idxs().tolist() == [1, 2, 3, 0]


class TestAddTransition:
    def test_forwards_fields(self, cls):
        calls = []
        buf = make(cls)
        buf.add = lambda *args: calls.append(args)
        tr = Transition(obs=1, action=2, reward=3, next_obs=4, terminated=5, truncated=6, extras={})
        buf.add_transition(tr)
        assert calls == [(1, 2, 3, 4, 5, 6)]


class TestSeed:
    def test_seeded_rng(self, cls):
        draw = lambda seed: make(cls, seed=seed).rng.integers(0, 1000, 8).tolist()
        assert draw(1) == draw(1) and draw(1) != draw(2)


def test_replay_sample_fields():
    assert ReplaySample._fields == ('obs', 'actions', 'rewards', 'next_obs', 'terminated',
                                    'truncated', 'idxs')


class TestStateDict:
    def filled(self, cls, n=3, **kw):
        buf = make(cls, **kw)
        tags = np.arange(n, dtype=np.float32)
        buf._store({'obs': np.repeat(tags[:, None], 4, 1), 'rewards': tags})
        return buf

    def test_contents(self, cls):
        state = self.filled(cls).state_dict()
        assert (state['kind'], state['maxlength'], state['ptr'], state['size']) == ('test', 5, 3, 3)
        assert state['specs']['obs'] == ((4,), '<f4') and state['specs']['terminated'] == ((), '|b1')
        assert set(state['data']) == set(FIELDS)

    def test_only_filled_rows_saved_as_cpu_tensors(self, cls):
        state = self.filled(cls, maxlength=5).state_dict()
        for name, rows in state['data'].items():
            assert isinstance(rows, torch.Tensor) and rows.device.type == 'cpu'
            assert rows.shape[0] == 3, name
        assert state['data']['rewards'].tolist() == [0, 1, 2]

    def test_loads_with_weights_only(self, cls, tmp_path):
        torch.save(self.filled(cls).state_dict(), tmp_path / 'buf.pt')
        buf = make(cls)
        buf.load_state_dict(torch.load(tmp_path / 'buf.pt', weights_only=True))
        assert len(buf) == 3 and buf.to_np(buf.rewards)[:3].tolist() == [0, 1, 2]

    def test_extra_specs_saved(self, cls):
        class Prioritized(cls):
            def _buffer_specs(self):
                return {**super()._buffer_specs(), 'priorities': ((), np.dtype(np.float64))}
        src = make(Prioritized)
        src._store({'priorities': np.array([0.5, 2.0]), 'rewards': np.zeros(2, np.float32)})
        dst = make(Prioritized)
        dst.load_state_dict(src.state_dict())
        assert dst.to_np(dst.priorities)[:2].tolist() == [0.5, 2.0]

    def test_maxlength_mismatch(self, cls):
        with pytest.raises(ValueError, match='maxlength'):
            make(cls, maxlength=6).load_state_dict(self.filled(cls).state_dict())

    @pytest.mark.parametrize('kw', [{'obs_dim': 5}, {'action_dim': 3}, {'obs_dtype': np.uint8},
                                    {'reward_dim': 2}])
    def test_layout_mismatch(self, cls, kw):
        with pytest.raises(ValueError, match='layout'):
            make(cls, **kw).load_state_dict(self.filled(cls).state_dict())

    @pytest.mark.parametrize('field,value', [('size', 9), ('ptr', 5), ('ptr', -1)])
    def test_invalid_counters(self, cls, field, value):
        state = {**self.filled(cls).state_dict(), field: value}
        with pytest.raises(ValueError, match='invalid state'):
            make(cls).load_state_dict(state)

    def test_row_count_mismatch(self, cls):
        state = self.filled(cls).state_dict()
        state['data']['obs'] = state['data']['obs'][:2]
        with pytest.raises(ValueError, match='obs'):
            make(cls).load_state_dict(state)

    def test_failed_load_leaves_buffer_untouched(self, cls):
        buf = self.filled(cls, n=2)
        state = self.filled(cls).state_dict()
        state['data']['obs'] = state['data']['obs'][:2]
        with pytest.raises(ValueError):
            buf.load_state_dict(state)
        assert len(buf) == 2 and buf.to_np(buf.rewards)[:2].tolist() == [0, 1]
