# Multi-agent RL

Every RL algorithm in dragon-torch has a multi-agent trainer alongside the single-agent one. Agents are trained in **groups**. Agents in a group share one network; different groups have separate networks. Each agent's critic sees only that agent's own observation, which makes this independent learning with parameter sharing (IPPO when used with PPO).

```python
from dragon_torch.data.envs import MultiAgentEnvConfig
from dragon_torch.models.RL import DragonRLConfig
from dragon_torch.trainers.RL import EvalConfig, MultiAgentPPOTrainer, MultiAgentPPOTrainerConfig

cfg = MultiAgentPPOTrainerConfig(
    env=MultiAgentEnvConfig(env_fn=my_parallel_env, n_envs=8),
    model=DragonRLConfig(checkpoint_dir='runs/ma-ppo', save_every=50, total_timesteps=200_000),
    eval=EvalConfig(every_steps=20_000, n_episodes=16))

with MultiAgentPPOTrainer(cfg) as trainer:
    history = trainer.fit()
    team_return = trainer.evaluate()
```

`MultiAgentA2CTrainer` / `MultiAgentA2CTrainerConfig` work the same way. The trainers train `MultiAgentPPOModel` / `MultiAgentA2CModel`, and the loss settings sit on the model config at the same paths as in single-agent runs: `model=MultiAgentPPOConfig(..., clip_coef=0.1)`, or `model.clip_coef` in `apply_overrides` and search spaces. The PPO and A2C [tuning presets](tuning.md#search-spaces) therefore apply as-is. A plain `DragonRLConfig` is converted to the algorithm's config, and a single-agent `PPOConfig` keeps its loss settings when converted. Every other config section (`rollout`, `eval`, `policy`, `optim`, `guard`) means the same as in [rl_trainers.md](rl_trainers.md).

## Layout

| Layer | File | Contents |
|---|---|---|
| Env | `dragon_torch/data/envs/multi_agent.py` | `MultiAgentEnvConfig`, `GymMultiAgentEnv`, `AgentGroups`, `MultiAgentVectorEnv`, `check_multi_agent_env_config` |
| Sources | `dragon_torch/data/RL/multi_agent.py` | `MultiAgentCollector`, `MultiAgentRolloutSource`, `MultiAgentEpisodeSource` |
| Network | `dragon_torch/modules/RL/policy.py` | `MultiAgentPolicyModule` (the base class) and `MultiAgentActorCritic` |
| Loss | `dragon_torch/loss/RL.py` | `MultiAgentLoss`, which applies any single-agent `RLLoss` to each group |
| Model | `dragon_torch/models/RL/multi_agent/` | `MultiAgentRLModel`, `MultiAgentRLConfig`, and the algorithm models `MultiAgentPPOModel` / `MultiAgentPPOConfig` and `MultiAgentA2CModel` / `MultiAgentA2CConfig` |
| Trainers | `dragon_torch/trainers/RL/` | `MultiAgentRLTrainer`, `MultiAgentPPOTrainer`, `MultiAgentA2CTrainer` and their configs |

## Envs

`MultiAgentEnvConfig` accepts two kinds of env.

**PettingZoo-style `ParallelEnv`.** Set `env_fn` to any callable that returns an object with the PettingZoo parallel API: `possible_agents`, `agents`, `observation_space(agent)`, `action_space(agent)`, `reset(seed, options) -> (obs, infos)` and `step(actions) -> (obs, rewards, terminations, truncations, infos)`, where every value is an `{agent: ...}` dict. The API is detected by duck typing, so PettingZoo itself doesn't need to be installed. A PettingZoo env such as `pettingzoo.mpe.simple_spread_v3.parallel_env` works directly.

**Gymnasium env with per-agent `Dict` spaces.** Set `env_fn` (or `env_id` for a registered env) to a gymnasium env whose `observation_space` and `action_space` are `Dict` spaces keyed by agent name. `GymMultiAgentEnv` adapts it. Rewards, `terminated` and `truncated` can each be:

- an `{agent: value}` dict,
- a sequence in agent order, or
- a scalar shared by every agent, such as a team reward or a common episode end.

All agents start every episode, and an agent leaves once its own `terminated` or `truncated` is set. Done agents are sent a placeholder action, because a `Dict` action space needs every key.

| Field | Default | Meaning |
|---|---|---|
| `env_id` / `env_fn`, `env_kwargs` | | How to build one env. Set exactly one of `env_id` and `env_fn` |
| `wrappers` | `[]` | Applied in order to the base env: gymnasium wrappers (`WrapperSpec`, `str` or callable) for gymnasium envs, callables such as PettingZoo/SuperSuit wrappers for `ParallelEnv`s |
| `groups` | `None` | Explicit `{group: [agents]}` |
| `share_params` | `True` | With the default grouping, `False` puts each agent in its own group |
| `team_return` | `'sum'` | How per-agent episode returns combine into the logged and evaluated return: `'sum'` or `'mean'` |
| `n_envs`, `seed` | `1`, `0` | Copies stepped per vector step, and the reset seed (env `i` gets `seed + i`) |

Vectorisation is synchronous only: `MultiAgentVectorEnv` steps its `n_envs` copies in turn.

## Agent groups

By default, agents are grouped by name prefix: `adversary_0` and `adversary_1` form the group `adversary`, and an agent with no numeric suffix is its own group.

- Agents in a group must have identical observation and action spaces, because they share a network. Otherwise building the env raises `ValueError`.
- Group names must be Python identifiers, because each one names a submodule.
- Every agent must be listed exactly once.

Inside the framework, all data is keyed by group, never by agent name. Within a group, agents are laid out along an agent axis: `obs[group]` has shape `[n_envs, agents_in_group, ...]`, and an `alive` mask marks which slots hold an agent that is currently in its env. Slots for agents that haven't spawned yet, or have already finished, hold placeholder observations and are left out of training.

## Episodes and returns

- An env's episode ends when all of its agents are done, and the env is reset in the same step. Any agent still acting when its env ends is treated as truncated, so its value bootstraps.
- An agent that leaves earlier is treated as terminated. GAE cuts at its own end, even though the env continues.
- `global_step` counts env steps (`n_envs` per vector step), as in single-agent runs, so `total_timesteps` means the same thing in both.

The history records the following keys:

| Key | Value |
|---|---|
| `episode_return`, `val_return` | Team return: the env's raw rewards, summed or averaged over agents (`team_return`) |
| `episode_return_shaped` | The same after `wrappers`, when wrappers are set. Raw rewards are recorded beneath your wrappers, as in single-agent runs |
| `episode_return/<group>`, `val_return/<group>` | The group's mean per-agent return |
| `policy_loss/<group>`, `entropy/<group>`, ... | Each group's loss components |
| `policy_loss`, `entropy`, ... | Means over groups. `approx_kl` and `clip_frac` take the maximum, so `DivergenceGuard` reacts to the worst group |

## Batches

`MultiAgentRolloutSource` minibatches over env steps, so all the agents from one step stay in the same minibatch. Within a minibatch, each group's live agents are flattened into one batch:

```python
{'obs': {group: [B_g, ...]},
 'targets': {group: {'actions', 'log_prob', 'values', 'advantages', 'returns'}}}
```

A group with no live agents in a minibatch is left out. `MultiAgentLoss` skips it.

## Custom networks

Pass `module=` with any `MultiAgentPolicyModule`: `forward({group: obs [B_g, ...]})` must return `{group: PolicyOutput}`. Per-group networks can be built with:

```python
MultiAgentActorCritic(policies={'red': ActorCritic(obs, act, encoder=cnn, features_dim=256),
                                'blue': ActorCritic(obs2, act2)})
```

To use a different loss per group, pass `MultiAgentLoss({'red': PPOLoss(...), 'blue': PPOLoss(...)}, weights={'blue': 0.5})`, for example from an overridden `build_loss`.

## Checking an env

With `validate_env=True`, the trainer builds one env from `env` (and from `eval.env`, if set) with its wrappers applied before building anything else. It checks the agent grouping, then runs a 20-step random rollout. It raises `EnvCheckError` if an observation falls outside its space, if any dict is keyed by an unknown agent, or if a reward is not finite. Call `check_multi_agent_env_config(cfg)` to run the same check directly.

## Not yet supported

- Centralised critics (MAPPO with a global state) and value-factorisation methods (QMIX and similar). To add one, write a `MultiAgentPolicyModule` and a matching source.
- Async vectorisation of multi-agent envs.
- Turn-based (AEC) PettingZoo envs. Convert them with `pettingzoo.utils.conversions.aec_to_parallel`.
