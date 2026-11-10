# RL components

The pieces a trainer assembles: models (`dragon_torch.models.RL`), policy networks (`dragon_torch.modules.RL`), losses (`dragon_torch.loss.RL`) and data sources (`dragon_torch.data.RL`). Use them directly to train without a [trainer](rl_trainers.md).

| | |
|---|---|
| [`PPOModel`, `A2CModel`](#ppomodel) | Algorithm models: a policy, plus the loss built from its config |
| [`DragonRLConfig`](#dragonrlconfig) | Settings shared by every RL model |
| [`DragonRLModel`](#dragonrlmodel) | Abstract base: acting, training steps, evaluation |
| [`ActorCritic`](#actorcritic) | Default policy network |
| [`PPOLoss`, `A2CLoss`, `MultiAgentLoss`](#ppoloss) | Policy-gradient losses |
| [`RolloutSource`](#rolloutsource) | On-policy training data with GAE |
| [`EpisodeSource`](#episodesource) | Evaluation episodes |

Every model, loss and source has a multi-agent counterpart, documented in [Multi-agent RL](multi_agent.md).

**Example**
```python
>>> env = DragonEnvConfig(env_id='CartPole-v1', n_envs=4)
>>> source = RolloutSource(env, n_steps=128)
>>> col = source.collector
>>> model = PPOModel(ActorCritic(col.observation_space, col.action_space),
...                  PPOConfig(checkpoint_dir='runs/ppo', save_every=10, total_timesteps=50_000))
>>> history = model.fit(source, torch.optim.Adam(model.parameters(), lr=3e-4))
>>> model.evaluate(EpisodeSource(env, n_episodes=10))
```

## Models

### PPOModel

```python
class dragon_torch.models.RL.PPOModel(module, cfg, loss_fn=None)
class dragon_torch.models.RL.A2CModel(module, cfg, loss_fn=None)
```

A [`SingleAgentRLModel`](#dragonrlmodel) whose loss is built from its config.

**Parameters**
- **module** (*SingleAgentPolicyModule*) – the policy, for example an [`ActorCritic`](#actorcritic).
- **cfg** (*PPOConfig* or *A2CConfig*) – [`DragonRLConfig`](#dragonrlconfig) plus the loss settings below. Any other config type raises `TypeError`.
- **loss_fn** (*callable, optional*) – replaces the loss the config would build.

| Config | `kind` | Loss settings (defaults) |
|---|---|---|
| `PPOConfig` | `'ppo-model'` | `clip_coef=0.2`, `vf_coef=0.5`, `ent_coef=0.0`, `clip_vloss=True`, `normalize_advantages=True` |
| `A2CConfig` | `'a2c-model'` | `vf_coef=0.5`, `ent_coef=0.0`, `normalize_advantages=False` |

The loss settings come from `PPOLossConfig` / `A2CLossConfig`, whose `build_loss()` makes the loss. Because the configs set `kind`, `model_factory` builds the matching class, and checkpoints are named after it.

> **Note**
> PPO expects several update epochs of several minibatches per rollout. A2C expects one: `RolloutSource(update_epochs=1, n_minibatches=1)`.

### DragonRLConfig

```python
class dragon_torch.models.RL.DragonRLConfig(checkpoint_dir, save_every, max_epochs=1, kind='rl-model',
                                            device='cpu', total_timesteps=None, max_grad_norm=0.5,
                                            eval_return='raw')
```

**Parameters**
- **checkpoint_dir** (*path*) – where checkpoints are written.
- **save_every** (*int*) – checkpoint every N epochs, and always after the last one.
- **max_epochs** (*int*) – epochs per `fit()` when neither `epochs` nor `total_timesteps` decides. Default: `1`.
- **device** (*str*) – Default: `'cpu'`.
- **total_timesteps** (*int, optional*) – env-step budget. `fit()` runs enough epochs to reach it, counting steps already taken, so a resumed run finishes the same budget.
- **max_grad_norm** (*float, optional*) – gradient-norm clip. `None` disables it. Default: `0.5`.
- **eval_return** (*str*) – `'raw'` scores the env's own reward and `'shaped'` scores the reward after wrappers; see [raw and shaped rewards](envs.md#raw-and-shaped-rewards). Default: `'raw'`.

### DragonRLModel

```python
class dragon_torch.models.RL.DragonRLModel(module, loss_fn, cfg)
class dragon_torch.models.RL.SingleAgentRLModel(module, loss_fn, cfg)
```

Abstract base of every RL model. `SingleAgentRLModel` implements it for Gymnasium envs, and `MultiAgentRLModel` for [multi-agent envs](multi_agent.md).

**Methods**
- **act**(*obs, deterministic=False*) → `ActOutput` – sample (or pick greedily) actions for a batch of obs; `extras` holds `log_prob` and `value`.
- **value**(*obs*) → *Tensor* – state values for a batch of obs.
- **fit**(*train_data, optimizer, val_data=None, epochs=None, callbacks=()*) → *list[dict]* – train and return the per-epoch history. Each step clips gradients and records the loss components.
- **evaluate**(*data*) → *float* – mean episode return over an [`EpisodeSource`](#episodesource). All of the means are kept in `last_eval`.

Class attributes: `higher_is_better = True` and `eval_metric = 'val_return'`.

## Networks

### ActorCritic

```python
class dragon_torch.modules.RL.ActorCritic(observation_space, action_space, hidden=(64, 64),
                                          activation=nn.Tanh, encoder=None, features_dim=None)
```

Actor-critic for on-policy algorithms. `forward(obs)` returns a `PolicyOutput(dist, value)`.

**Parameters**
- **observation_space**, **action_space** (*gym.Space*) – the spaces after wrapping.
- **hidden** (*tuple of int*) – MLP layer sizes. Default: `(64, 64)`.
- **activation** (*type*) – Default: `nn.Tanh`.
- **encoder** (*nn.Module, optional*) – a feature extractor shared by actor and critic, such as a CNN for pixels. Default: separate MLP trunks over flattened obs.
- **features_dim** (*int, optional*) – the encoder's output size. Default: `encoder.features_dim`.

Without an encoder, `Box` and `Discrete` observations are supported. The action distribution is picked from the action space:

| Action space | Distribution |
|---|---|
| `Discrete` | Categorical |
| `MultiDiscrete` (1-D) | Independent categoricals |
| `MultiBinary` | Bernoulli |
| `Box` | Diagonal Gaussian with a learned log-std. Samples are clipped to the bounds when stepping the env |

Custom networks subclass `SingleAgentPolicyModule` and return a `PolicyOutput` from `forward`.

## Losses

### PPOLoss

```python
class dragon_torch.loss.RL.PPOLoss(clip_coef=0.2, vf_coef=0.5, ent_coef=0.0, clip_vloss=True,
                                   normalize_advantages=True)
class dragon_torch.loss.RL.A2CLoss(vf_coef=0.5, ent_coef=0.0, normalize_advantages=False)
class dragon_torch.loss.RL.MultiAgentLoss(loss, weights=None)
```

`loss(policy_output, targets)` returns a scalar, and the components are left in `last_metrics`, which the model averages into the epoch history:

| Loss | Recorded |
|---|---|
| `PPOLoss` | `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_frac` |
| `A2CLoss` | `policy_loss`, `value_loss`, `entropy` |

`MultiAgentLoss` applies a single-agent loss to each agent group; see [Multi-agent RL](multi_agent.md#multiagentloss).

## Data

### RolloutSource

```python
class dragon_torch.data.RL.RolloutSource(env_cfg, n_steps=128, gamma=0.99, gae_lambda=0.95,
                                         update_epochs=4, n_minibatches=4, normalize_obs=False,
                                         normalize_reward=False)
```

On-policy training data. Each iteration (one epoch) collects `n_steps × n_envs` transitions with the bound model, computes GAE, and yields `update_epochs × n_minibatches` shuffled minibatches:

```python
{'obs': ..., 'targets': {'actions', 'log_prob', 'values', 'advantages', 'returns'}}
```

**Parameters**
- **env_cfg** (*DragonEnvConfig*) – the training env.
- **n_steps** (*int*) – steps per sub-env per rollout.
- **gamma**, **gae_lambda** (*float*) – discount and GAE λ.
- **update_epochs**, **n_minibatches** (*int*) – passes over each rollout, and minibatches per pass.
- **normalize_obs**, **normalize_reward** (*bool*) – running normalisation; see [Normalisation](envs.md#normalisation).

Truncated episodes bootstrap from the true final obs, and terminated ones don't. The source's state (step count, normalisers, RNG) is saved in checkpoints. `metrics()` adds `global_step`, `steps_per_sec`, `episode_return` and `episode_length` to each epoch's record.

### EpisodeSource

```python
class dragon_torch.data.RL.EpisodeSource(env_cfg, n_episodes=10, deterministic=True, max_steps=1_000_000)
```

Evaluation data. Iterating runs the bound model until `n_episodes` finish, yielding `{'return', 'shaped_return', 'length'}` for each. Every iteration re-seeds the env, so successive evaluations are comparable. The source borrows the training source's observation normaliser without updating it. `max_steps` caps the vector steps per evaluation, so an env that never ends raises instead of hanging.

> **Note**
> Both sources step the env through an `EnvCollector`, which calls the model's `act` / `value` (the `Actor` and `ValueActor` protocols). Any object with those methods can collect data.
