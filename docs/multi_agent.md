# Multi-agent RL

Every algorithm has a multi-agent version. Agents are trained in **groups**: agents in a group share one network, and each group has its own. Each critic sees only its own agent's observation, so this is independent learning with parameter sharing (IPPO, with PPO).

**Example**
```python
>>> cfg = MultiAgentPPOTrainerConfig(
...     env=MultiAgentEnvConfig(env_fn=make_parallel_env, n_envs=8),  # returns a ParallelEnv
...     model=MultiAgentPPOConfig(checkpoint_dir='runs/ma-ppo', save_every=50,
...                               total_timesteps=200_000, clip_coef=0.1),
...     eval=EvalConfig(every_steps=20_000, n_episodes=16))
>>> with MultiAgentPPOTrainer(cfg) as trainer:
...     history = trainer.fit()
```

Configs, trainers and tuning presets work as in the single-agent case, at the same config paths; see [RL trainers](rl_trainers.md).

| Single-agent | Multi-agent |
|---|---|
| `DragonEnvConfig` | [`MultiAgentEnvConfig`](#multiagentenvconfig) |
| `RolloutSource`, `EpisodeSource` | `MultiAgentRolloutSource`, `MultiAgentEpisodeSource` (same parameters) |
| `ActorCritic` | [`MultiAgentActorCritic`](#multiagentactorcritic) |
| `PPOLoss`, `A2CLoss` | [`MultiAgentLoss`](#multiagentloss) around either |
| `PPOModel`, `A2CModel` | `MultiAgentPPOModel`, `MultiAgentA2CModel` |
| `PPOConfig`, `A2CConfig` | `MultiAgentPPOConfig`, `MultiAgentA2CConfig` (same fields, kinds `'marl-ppo-model'` and `'marl-a2c-model'`) |
| `PPOTrainer`, `A2CTrainer` | `MultiAgentPPOTrainer`, `MultiAgentA2CTrainer` |

## MultiAgentEnvConfig

```python
class dragon_torch.data.envs.MultiAgentEnvConfig(env_id=None, env_fn=None, env_kwargs={}, wrappers=[],
                                                 groups=None, share_params=True, team_return='sum',
                                                 n_envs=1, seed=0)
```

**Parameters**
- **env_id**, **env_fn** (*optional*) – set exactly one. The env must be either:
  - a **PettingZoo-style `ParallelEnv`**: `possible_agents`, `agents`, `observation_space(agent)`, `action_space(agent)`, `reset()` and `step()` over `{agent: value}` dicts. It's detected by duck typing, so PettingZoo itself is optional.
  - a **Gymnasium env with `Dict` spaces keyed by agent**. Its reward, `terminated` and `truncated` may each be an `{agent: value}` dict, a sequence in agent order, or a scalar shared by every agent.
- **env_kwargs** (*dict*) – keyword arguments for the env.
- **wrappers** (*list*) – applied in order to the base env: Gymnasium wrappers for Gymnasium envs, or callables (for example SuperSuit wrappers) for `ParallelEnv`s.
- **groups** (*dict, optional*) – `{group: [agents]}`. Default: group by name prefix, so `adversary_0` and `adversary_1` form `adversary`.
- **share_params** (*bool*) – with the default grouping, `False` gives each agent its own group. Default: `True`.
- **team_return** (*str*) – `'sum'` or `'mean'`: how per-agent returns combine into the logged and evaluated return. Default: `'sum'`.
- **n_envs**, **seed** (*int*) – copies stepped per vector step, and the reset seed (env `i` gets `seed + i`).

Agents in a group must have identical spaces, and group names must be Python identifiers.

Related:
- `AgentGroups.from_env(env, groups=None, share_params=True)` resolves and validates the grouping.
- `MultiAgentVectorEnv(cfg)` steps the copies synchronously.
- `check_multi_agent_env_config(cfg)` builds one env and runs a 20-step random rollout, raising `EnvCheckError` on bad obs, unknown agents or non-finite rewards. Trainers run it when `validate_env=True`.

## MultiAgentActorCritic

```python
class dragon_torch.modules.RL.MultiAgentActorCritic(spaces_by_group=None, policies=None, **actor_critic_kw)
```

One [`ActorCritic`](rl_models.md#actorcritic) per group. `forward({group: obs [B_g, ...]})` returns `{group: PolicyOutput}`.

**Parameters**
- **spaces_by_group** (*dict, optional*) – `{group: (obs_space, action_space)}`, each built into an `ActorCritic(**actor_critic_kw)`.
- **policies** (*dict, optional*) – `{group: SingleAgentPolicyModule}`, for custom per-group networks. Pass exactly one of `spaces_by_group` and `policies`.

```python
>>> MultiAgentActorCritic(policies={'red': ActorCritic(obs, act, encoder=cnn, features_dim=256),
...                                 'blue': ActorCritic(obs2, act2)})
```

Custom networks subclass `MultiAgentPolicyModule` and are passed as `module=`.

## MultiAgentLoss

```python
class dragon_torch.loss.RL.MultiAgentLoss(loss, weights=None)
```

Applies a single-agent loss to each group and sums the results.

**Parameters**
- **loss** (*RLLoss or dict*) – one loss for every group (copied per group), or `{group: loss}`.
- **weights** (*dict, optional*) – per-group weights. Default: `1.0`.

`last_metrics` holds `'<name>/<group>'` for each group, plus `'<name>'` aggregated over groups. The aggregate is the mean, except for `approx_kl` and `clip_frac`, which take the maximum so that `DivergenceGuard` reacts to the worst group. Groups missing from a minibatch are skipped.

## Notes

### Data layout

All data is keyed by group, never by agent. Within a group, agents sit on an agent axis: `obs[group]` has shape `[n_envs, agents_in_group, ...]`, and an `alive` mask marks the slots whose agent is currently in its env. Empty slots hold placeholder obs and are left out of training.

`MultiAgentRolloutSource` minibatches over env steps, so one step's agents stay together. It then flattens each group's live agents:

```python
{'obs': {group: [B_g, ...]}, 'targets': {group: {'actions', 'log_prob', 'values', 'advantages', 'returns'}}}
```

### Episodes

- An env's episode ends when all of its agents are done, and the env resets in the same step. Agents still acting at that point are truncated, so their values bootstrap.
- An agent that leaves earlier is terminated, and GAE cuts at its own end.
- `global_step` counts env steps, as in single-agent runs.

### History keys

| Key | Value |
|---|---|
| `episode_return`, `val_return` | Team return (`team_return` over agents), from raw rewards |
| `episode_return_shaped` | The same after `wrappers`, when wrappers are set |
| `episode_return/<group>`, `val_return/<group>` | The group's mean per-agent return |
| `policy_loss/<group>`, ... | Per-group loss components |

### Not supported

- Centralised critics (MAPPO) and value factorisation (QMIX).
- Async vectorisation.
- Turn-based (AEC) PettingZoo envs. Convert them with `pettingzoo.utils.conversions.aec_to_parallel`.
