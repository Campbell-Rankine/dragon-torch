"""Multi-agent envs: PettingZoo-style ParallelEnvs and gymnasium envs with
per-agent Dict spaces, behind one interface.

Internally every multi-agent env is a *ParallelEnv* (the PettingZoo parallel
API, duck-typed so PettingZoo itself is optional):
    possible_agents, agents, observation_space(agent), action_space(agent),
    reset(seed, options) -> (obs, infos), step(actions) -> (obs, rewards,
    terminations, truncations, infos), every value a {agent: ...} dict.

A gymnasium env is adapted by GymMultiAgentEnv when its observation and action
spaces are Dicts keyed by agent. Its reward / terminated / truncated may be
{agent: value} dicts, sequences ordered like the agents, or scalars shared by
every agent (a team reward, a common episode end).

Agents are trained in groups (AgentGroups): agents of a group share a network
and so must have identical spaces. MultiAgentVectorEnv steps n_envs copies and
packs everything per group as [n_envs, agents_in_group, ...] arrays, padding
agents that aren't currently in an env (not yet spawned, or already done) with
placeholder obs and an `alive` mask of False.
"""
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .config import EnvCheckError
from .registry import WrapperSpec
from .wrappers import RAW_REWARD_KEY

type ParallelEnv = Any
type MultiAgentEnvFn = Callable[..., Union[ParallelEnv, gym.Env]]
type MultiAgentWrapperFn = Callable[[Any], Any]

_PARALLEL_API = ('possible_agents', 'agents', 'observation_space', 'action_space', 'reset', 'step')


def is_parallel_env(env: Any) -> bool:
    """Whether `env` follows the PettingZoo parallel API (by duck typing)."""
    return (not isinstance(env, gym.Env) and all(hasattr(env, a) for a in _PARALLEL_API)
            and callable(env.observation_space))


# ------- placeholders ------- #
def placeholder(space: spaces.Space) -> Any:
    """A fixed, valid value of `space`: padding for absent agents' obs, and the
    action sent for done agents to envs that still want one."""
    if isinstance(space, spaces.Box):
        return np.clip(np.zeros(space.shape, space.dtype), space.low, space.high).astype(space.dtype)
    if isinstance(space, spaces.Discrete):
        return np.asarray(space.start, space.dtype)
    if isinstance(space, spaces.MultiDiscrete):
        return np.asarray(space.start, space.dtype).copy()
    if isinstance(space, spaces.MultiBinary):
        return np.zeros(space.shape, space.dtype)
    if isinstance(space, spaces.Dict):
        return {k: placeholder(s) for k, s in space.spaces.items()}
    if isinstance(space, spaces.Tuple):
        return tuple(placeholder(s) for s in space.spaces)
    return space.sample()


def np_tree_stack(items: Sequence[Any]) -> Any:
    first = items[0]
    if isinstance(first, dict):
        return {k: np_tree_stack([it[k] for it in items]) for k in first}
    if isinstance(first, tuple):
        return tuple(np_tree_stack([it[i] for it in items]) for i in range(len(first)))
    return np.stack([np.asarray(it) for it in items])


# ------- raw reward recording ------- #
class RecordRawMultiAgentReward(gym.Wrapper):
    """Gymnasium multi-agent env: copies each step's reward (dict / sequence /
    scalar) into info['raw_reward'] before outer wrappers change it."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info[RAW_REWARD_KEY] = dict(reward) if isinstance(reward, Mapping) else np.copy(reward)
        return obs, reward, terminated, truncated, info


class RecordRawParallelReward:
    """ParallelEnv: puts each agent's reward in infos[agent]['raw_reward']
    before outer (e.g. PettingZoo / SuperSuit) wrappers change it. Everything
    else is delegated to the wrapped env."""

    def __init__(self, env: ParallelEnv):
        self.env = env

    def __getattr__(self, name: str) -> Any:
        if name == 'env':
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, 'unwrapped', self.env)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions: dict[str, Any]):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        infos = {a: dict(infos.get(a, {})) for a in set(infos) | set(rewards)}
        for agent, r in rewards.items():
            infos[agent][RAW_REWARD_KEY] = float(r)
        return obs, rewards, terms, truncs, infos

    def close(self) -> None:
        close = getattr(self.env, 'close', None)
        if callable(close):
            close()


# ------- gymnasium adapter ------- #
class GymMultiAgentEnv:
    """A gymnasium env with Dict(agent -> space) observation and action spaces,
    as a ParallelEnv. All agents start every episode; an agent leaves once its
    terminated / truncated is set, and done agents are sent a placeholder
    action (a Dict action space needs every key)."""

    def __init__(self, env: gym.Env, agents: Optional[Sequence[str]] = None):
        obs_space, act_space = env.observation_space, env.action_space
        if not isinstance(obs_space, spaces.Dict) or not isinstance(act_space, spaces.Dict):
            raise TypeError("a gymnasium multi-agent env needs Dict observation and action spaces "
                            f"keyed by agent, got {obs_space} and {act_space}")
        if set(obs_space.spaces) != set(act_space.spaces):
            raise ValueError(f"observation agents {sorted(obs_space.spaces)} != "
                             f"action agents {sorted(act_space.spaces)}")
        self.env = env
        self.possible_agents = [str(a) for a in (agents or act_space.spaces)]
        self.agents: list[str] = []
        self.metadata = getattr(env, 'metadata', {})
        self._noop = {a: placeholder(act_space[a]) for a in self.possible_agents}

    def observation_space(self, agent: str) -> spaces.Space:
        return self.env.observation_space[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.env.action_space[agent]

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def _per_agent(self, value: Any, name: str) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return {a: value[a] for a in self.possible_agents if a in value}
        arr = np.asarray(value)
        if arr.ndim == 0:
            return {a: arr.item() for a in self.possible_agents}
        if arr.shape[0] != len(self.possible_agents):
            raise ValueError(f"{name} has {arr.shape[0]} entries for {len(self.possible_agents)} "
                             "agents; return a {agent: value} dict, one value per agent, or a scalar")
        return {a: arr[i].item() for i, a in enumerate(self.possible_agents)}

    def _infos(self, info: dict[str, Any]) -> dict[str, dict]:
        if info and set(info) <= set(self.possible_agents):
            return {a: dict(info.get(a, {})) for a in self.possible_agents}
        common = {k: v for k, v in info.items() if k != RAW_REWARD_KEY}
        infos = {a: dict(common) for a in self.possible_agents}
        if RAW_REWARD_KEY in info:
            for a, r in self._per_agent(info[RAW_REWARD_KEY], 'raw reward').items():
                infos[a][RAW_REWARD_KEY] = float(r)
        return infos

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)
        return {a: obs[a] for a in self.agents}, self._infos(info)

    def step(self, actions: Mapping[str, Any]):
        acting = list(self.agents)
        full = {a: actions[a] if a in acting and a in actions else self._noop[a]
                for a in self.possible_agents}
        obs, reward, terminated, truncated, info = self.env.step(full)
        rewards = {a: float(r) for a, r in self._per_agent(reward, 'reward').items() if a in acting}
        terms = {a: bool(v) for a, v in self._per_agent(terminated, 'terminated').items() if a in acting}
        truncs = {a: bool(v) for a, v in self._per_agent(truncated, 'truncated').items() if a in acting}
        infos = {a: i for a, i in self._infos(info).items() if a in acting}
        self.agents = [a for a in acting if not (terms.get(a) or truncs.get(a))]
        return {a: obs[a] for a in acting}, rewards, terms, truncs, infos

    def close(self) -> None:
        self.env.close()


# ------- config ------- #
class MultiAgentEnvConfig(BaseModel):
    """How to build a vectorised multi-agent env.

    `env_fn(**env_kwargs)` returns a PettingZoo-style ParallelEnv or a gymnasium
    env with per-agent Dict spaces; `env_id` names a registered gymnasium env
    of the latter kind. `wrappers` are applied in order to the base env, so
    they're gymnasium wrappers (WrapperSpec / str / callable) for gymnasium
    envs and callables (e.g. PettingZoo / SuperSuit wrappers) for ParallelEnvs.

    Groups (agents sharing one network) default to agent-name prefixes:
    'adversary_0', 'adversary_1' -> 'adversary'. share_params=False puts each
    agent in its own group; `groups` sets them explicitly.
    `team_return` is how per-agent episode returns combine into the episode
    return that's logged and evaluated: their sum, or their mean.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    env_id: Optional[str] = None
    env_fn: Optional[MultiAgentEnvFn] = None
    env_kwargs: dict[str, Any] = {}
    wrappers: list[Union[WrapperSpec, MultiAgentWrapperFn]] = []
    groups: Optional[dict[str, list[str]]] = None
    share_params: bool = True
    team_return: Literal['sum', 'mean'] = 'sum'
    n_envs: int = Field(default=1, ge=1)
    seed: int = 0

    @model_validator(mode='after')
    def _one_source(self) -> "MultiAgentEnvConfig":
        if (self.env_id is None) == (self.env_fn is None):
            raise ValueError("Set exactly one of env_id or env_fn")
        if self.groups is not None and not self.share_params:
            raise ValueError("share_params=False only applies to the default grouping; "
                             "with explicit groups, give each agent its own group instead")
        return self

    @property
    def shaped(self) -> bool:
        """Whether wrappers sit between the env and the agents."""
        return bool(self.wrappers)

    @property
    def portable(self) -> bool:
        return self.env_fn is None and all(isinstance(w, WrapperSpec) for w in self.wrappers)


def make_multi_agent_env(cfg: MultiAgentEnvConfig) -> ParallelEnv:
    """One env as a ParallelEnv: base env -> raw-reward recorder -> cfg.wrappers
    -> (GymMultiAgentEnv for gymnasium envs)."""
    base = gym.make(cfg.env_id, **cfg.env_kwargs) if cfg.env_id is not None \
        else cfg.env_fn(**cfg.env_kwargs)
    if isinstance(base, gym.Env):
        env = RecordRawMultiAgentReward(base)
    elif is_parallel_env(base):
        env = RecordRawParallelReward(base)
    else:
        raise TypeError(f"multi-agent env must be a ParallelEnv (PettingZoo parallel API) or a "
                        f"gymnasium env with Dict spaces, got {type(base).__name__}")
    for wrapper in cfg.wrappers:
        env = wrapper(env)
    return GymMultiAgentEnv(env) if isinstance(env, gym.Env) else env


# ------- groups ------- #
_GROUP_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def default_group(agent: str) -> str:
    """'adversary_0' -> 'adversary'; names without a numeric suffix are their own group."""
    name = re.sub(r'[_\-]?\d+$', '', agent) or agent
    return re.sub(r'\W', '_', name)


@dataclass
class AgentGroups:
    """Which agents share a network: group -> agents (in a fixed order, which
    is their index along a group's agent axis), plus each group's spaces."""
    agents_by_group: dict[str, list[str]]
    observation_spaces: dict[str, spaces.Space]
    action_spaces: dict[str, spaces.Space]

    @classmethod
    def from_env(cls, env: ParallelEnv, groups: Optional[Mapping[str, Sequence[str]]] = None,
                 share_params: bool = True) -> "AgentGroups":
        agents = list(env.possible_agents)
        if groups is None:
            groups = {}
            for a in agents:
                name = default_group(a) if share_params else re.sub(r'\W', '_', a)
                groups.setdefault(name, []).append(a)
        groups = {g: list(members) for g, members in groups.items()}
        for g in groups:
            if not _GROUP_NAME.match(g):
                raise ValueError(f"group name {g!r} must be a Python identifier (it names a module)")
        listed = [a for members in groups.values() for a in members]
        dupes = sorted({a for a in listed if listed.count(a) > 1})
        missing, unknown = sorted(set(agents) - set(listed)), sorted(set(listed) - set(agents))
        if dupes or missing or unknown:
            raise ValueError(f"groups must list every agent exactly once: duplicated {dupes}, "
                             f"missing {missing}, unknown {unknown}")
        obs_spaces, act_spaces = {}, {}
        for g, members in groups.items():
            if not members:
                raise ValueError(f"group '{g}' has no agents")
            obs, act = env.observation_space(members[0]), env.action_space(members[0])
            for a in members[1:]:
                if env.observation_space(a) != obs or env.action_space(a) != act:
                    raise ValueError(f"agents in group '{g}' share a network so need identical "
                                     f"spaces, but {a} differs from {members[0]}")
            obs_spaces[g], act_spaces[g] = obs, act
        return cls(groups, obs_spaces, act_spaces)

    @property
    def names(self) -> list[str]:
        return list(self.agents_by_group)

    @property
    def agents(self) -> list[str]:
        return [a for members in self.agents_by_group.values() for a in members]

    def size(self, group: str) -> int:
        return len(self.agents_by_group[group])

    def index(self) -> dict[str, tuple[str, int]]:
        """agent -> (group, position along the group's agent axis)."""
        return {a: (g, j) for g, members in self.agents_by_group.items()
                for j, a in enumerate(members)}

    def spaces_by_group(self) -> dict[str, tuple[spaces.Space, spaces.Space]]:
        return {g: (self.observation_spaces[g], self.action_spaces[g]) for g in self.names}


# ------- vectorisation ------- #
@dataclass
class MultiAgentStep:
    """One vector step, every field a {group: [n_envs, agents_in_group, ...]} array.

    next_obs    obs to act on next (reset obs where env_done; placeholders for
                agents not in their env)
    next_alive  which agents next_obs is real for
    final_obs   true successor obs of this step's actions (what to bootstrap from)
    reward / raw_reward / terminated / truncated
                for agents that acted (zeros / False elsewhere). Agents still
                acting when their env ended are truncated, so they bootstrap.
    env_done    [n_envs] envs whose episode ended (and were reset)
    """
    next_obs: dict[str, Any]
    next_alive: dict[str, np.ndarray]
    final_obs: dict[str, Any]
    reward: dict[str, np.ndarray]
    raw_reward: dict[str, np.ndarray]
    terminated: dict[str, np.ndarray]
    truncated: dict[str, np.ndarray]
    env_done: np.ndarray
    infos: list[dict[str, dict]]


class MultiAgentVectorEnv:
    """n_envs copies of a multi-agent env, stepped in turn (sync), with
    same-step autoreset: an env whose agents are all done is reset at once and
    its reset obs returned as next_obs, while final_obs keeps the true last obs."""

    def __init__(self, cfg: MultiAgentEnvConfig):
        self.cfg = cfg
        self.envs = [make_multi_agent_env(cfg) for _ in range(cfg.n_envs)]
        self.num_envs = cfg.n_envs
        self.groups = AgentGroups.from_env(self.envs[0], cfg.groups, cfg.share_params)
        self._index = self.groups.index()
        self._obs_pad = {g: placeholder(s) for g, s in self.groups.observation_spaces.items()}

    def _pack_obs(self, per_env: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {g: np_tree_stack([np_tree_stack([obs[a] if a in obs else self._obs_pad[g]
                                                 for a in members]) for obs in per_env])
                for g, members in self.groups.agents_by_group.items()}

    def _zeros(self, dtype: Any) -> dict[str, np.ndarray]:
        return {g: np.zeros((self.num_envs, self.groups.size(g)), dtype) for g in self.groups.names}

    def _alive(self) -> dict[str, np.ndarray]:
        alive = self._zeros(bool)
        for i, env in enumerate(self.envs):
            for a in env.agents:
                g, j = self._index[a]
                alive[g][i, j] = True
        return alive

    def reset(self, seed: Optional[int] = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        obs = [env.reset(seed=None if seed is None else seed + i)[0]
               for i, env in enumerate(self.envs)]
        return self._pack_obs(obs), self._alive()

    def step(self, actions: Mapping[str, np.ndarray]) -> MultiAgentStep:
        reward, raw, term, trunc = self._zeros(np.float64), self._zeros(np.float64), \
            self._zeros(bool), self._zeros(bool)
        env_done = np.zeros(self.num_envs, bool)
        next_obs, final_obs, infos = [], [], []
        for i, env in enumerate(self.envs):
            acting = list(env.agents)
            obs, rewards, terms, truncs, info = env.step(
                {a: actions[self._index[a][0]][i, self._index[a][1]] for a in acting})
            remaining = set(env.agents)
            done_env = not remaining
            for a in acting:
                g, j = self._index[a]
                reward[g][i, j] = rewards.get(a, 0.0)
                raw[g][i, j] = info.get(a, {}).get(RAW_REWARD_KEY, reward[g][i, j])
                term[g][i, j] = bool(terms.get(a, False))
                trunc[g][i, j] = bool(truncs.get(a, False))
                if not (term[g][i, j] or trunc[g][i, j]):
                    if a not in remaining:      # removed without a flag: it's over for it
                        term[g][i, j] = True
                    elif done_env:              # env ended under it: bootstrap
                        trunc[g][i, j] = True
            final_obs.append(obs)
            infos.append(info)
            if done_env:
                env_done[i] = True
                obs, _ = env.reset()
            next_obs.append({a: obs[a] for a in env.agents if a in obs})
        return MultiAgentStep(next_obs=self._pack_obs(next_obs), next_alive=self._alive(),
                              final_obs=self._pack_obs(final_obs), reward=reward, raw_reward=raw,
                              terminated=term, truncated=trunc, env_done=env_done, infos=infos)

    def close(self) -> None:
        for env in self.envs:
            close = getattr(env, 'close', None)
            if callable(close):
                close()


def check_multi_agent_env_config(cfg: MultiAgentEnvConfig, name: str = 'env',
                                 n_steps: int = 20) -> None:
    """Build one env (wrappers applied), group its agents, and run a short
    random rollout, checking that obs match their spaces and that every dict is
    keyed by known agents. Raises EnvCheckError naming `name`."""
    env = None
    try:
        env = make_multi_agent_env(cfg)
        AgentGroups.from_env(env, cfg.groups, cfg.share_params)
        known = set(env.possible_agents)

        def check_obs(obs: Mapping[str, Any], when: str) -> None:
            if set(obs) - known:
                raise ValueError(f"{when} returned obs for unknown agents {sorted(set(obs) - known)}")
            for a, o in obs.items():
                if not env.observation_space(a).contains(o):
                    raise ValueError(f"{when}: obs of {a} is outside its observation_space")

        obs, _ = env.reset(seed=cfg.seed)
        check_obs(obs, 'reset()')
        for _ in range(n_steps):
            if not env.agents:
                obs, _ = env.reset()
                check_obs(obs, 'reset()')
            acts = {a: env.action_space(a).sample() for a in env.agents}
            obs, rewards, terms, truncs, _ = env.step(acts)
            check_obs(obs, 'step()')
            for label, d in (('rewards', rewards), ('terminations', terms), ('truncations', truncs)):
                if set(d) - known:
                    raise ValueError(f"step() {label} has unknown agents {sorted(set(d) - known)}")
            if not all(np.isfinite(float(r)) for r in rewards.values()):
                raise ValueError("step() returned a non-finite reward")
    except Exception as e:
        raise EnvCheckError(f"{name} failed the multi-agent env check: "
                            f"{type(e).__name__}: {e}") from e
    finally:
        close = getattr(env, 'close', None)
        if callable(close):
            close()
