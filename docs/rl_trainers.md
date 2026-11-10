# dragon_torch.trainers.RL

Build and run a whole on-policy RL run from one config: the rollout source, evaluator, policy network, loss, model, optimizer, LR annealing, step-scheduled evaluation and divergence guard. The plain `dragon_torch.trainers` package doesn't import this one, so supervised code doesn't need the `RL` extra.

| Algorithm | Single-agent | Multi-agent |
|---|---|---|
| PPO | `PPOTrainer(PPOTrainerConfig)` | `MultiAgentPPOTrainer(MultiAgentPPOTrainerConfig)` |
| A2C | `A2CTrainer(A2CTrainerConfig)` | `MultiAgentA2CTrainer(MultiAgentA2CTrainerConfig)` |

**Example**
```python
>>> cfg = PPOTrainerConfig(
...     env=DragonEnvConfig(env_id='CartPole-v1', n_envs=4),
...     model=PPOConfig(checkpoint_dir='runs/ppo', save_every=50, total_timesteps=25_000,
...                     ent_coef=0.01),
...     eval=EvalConfig(every_steps=5_000, n_episodes=8),
...     optim=OptimConfig(lr=2.5e-3, anneal_lr=True))
>>> with PPOTrainer(cfg) as trainer:     # closes the envs on exit
...     history = trainer.fit()          # until model.total_timesteps
...     score = trainer.evaluate()
```

## Trainers

### DragonRLTrainer

```python
class dragon_torch.trainers.RL.DragonRLTrainer(cfg, module=None)
```

Abstract base. A concrete trainer is an env kind plus an algorithm:

- **The env kind.** `SingleAgentRLTrainer` or `MultiAgentRLTrainer` builds the source, evaluator and network.
- **The algorithm.** `model_cls`, such as `PPOModel` or `MultiAgentPPOModel`, whose config holds the loss settings.

**Parameters**
- **cfg** (*DragonRLTrainerConfig*) – the run. It must be an instance of the trainer's `config_cls`, otherwise `TypeError`.
- **module** (*DragonPolicyModule, optional*) – the policy network. Default: built from `cfg.policy`.

**Methods**
- **fit**(*train_data=None, val_data=None, epochs=None, callbacks=()*) → *list[dict]* – train on the trainer's own source until `model.total_timesteps`, or for `epochs`. Default callbacks (eval, then guard) run before yours.
- **evaluate**(*data=None*) → *float* – mean return on the trainer's evaluator.
- **resume**(*path=None*) – load a checkpoint; see [Resuming](#resuming).
- **close**() – close the envs.

**Class attributes**
- **config_cls** – the config type accepted.
- **model_cls** – the model class built.
- **objectives** – the default tuning objective, `{'val_return_smooth': 'maximize'}`.

**Build hooks**, called in order. Override any of them.

| Hook | Single-agent | Multi-agent |
|---|---|---|
| `check_envs` | Gymnasium's checker, when `validate_env` is set | A short random rollout, when `validate_env` is set |
| `build_source` | `RolloutSource` | `MultiAgentRolloutSource` |
| `build_evaluator` | `EpisodeSource`, or `None` | `MultiAgentEpisodeSource`, or `None` |
| `build_module` | `ActorCritic` | `MultiAgentActorCritic` |
| `build_agent_loss` | `cfg.model.build_loss()` | same |
| `build_loss` | the agent loss | `MultiAgentLoss(agent loss)` |
| `build_model` | `model_cls(module=..., loss_fn=..., cfg=cfg.model)` | same |
| `build_optimizer`, `build_lr_schedule` | from `optim` | same |
| `build_eval_callback`, `build_guard` | from `eval`, `guard` | same |

Torch is seeded from `cfg.seed` before building. If a hook fails, the envs already opened are closed.

## Configs

### DragonRLTrainerConfig

```python
class dragon_torch.trainers.RL.DragonRLTrainerConfig(env, model, rollout=RolloutConfig(), eval=None,
                                                     policy=PolicyConfig(), optim=OptimConfig(),
                                                     guard=GuardConfig(), seed=0, validate_env=False,
                                                     epochs=None, resume=False)
```

Sections shared by every run. `SingleAgentRLTrainerConfig` sets `env` to a `DragonEnvConfig`, and `MultiAgentRLTrainerConfig` sets it to a `MultiAgentEnvConfig`.

**Parameters**
- **env** – the env to train on; see [Environments](envs.md).
- **model** (*DragonRLConfig*) – narrowed by each algorithm config to its model config (see below).
- **rollout** ([*RolloutConfig*](#rolloutconfig)) – rollout and minibatch settings.
- **eval** ([*EvalConfig*](#evalconfig)*, optional*) – step-scheduled evaluation. `None` disables it.
- **policy** ([*PolicyConfig*](#policyconfig)) – the default network.
- **optim** ([*OptimConfig*](#optimconfig)) – optimizer and LR annealing.
- **guard** ([*GuardConfig*](#guardconfig)*, optional*) – divergence guard. `None` disables it.
- **seed** (*int*) – torch seed for network initialisation.
- **validate_env** (*bool*) – check the wrapped env before building anything.
- **epochs**, **resume** – from `DragonTrainerConfig`.

**Validation**
- `rollout.n_steps × env.n_envs` must be divisible by `rollout.n_minibatches`.
- `optim.anneal_lr` requires `model.total_timesteps`.
- The config warns if `eval.every_steps` is less than one epoch of steps.
- The config warns if `guard.max_kl` is set but there's only one update per rollout; see [DivergenceGuard](#divergenceguard).

### Algorithm configs

| Config | `model` | Defaults that differ |
|---|---|---|
| `PPOTrainerConfig` | `PPOConfig` | `rollout`: 4 update epochs × 4 minibatches |
| `A2CTrainerConfig` | `A2CConfig` | `rollout`: `n_steps=5`, `gae_lambda=1.0`, 1 × 1. `optim`: RMSprop, `lr=7e-4`. `guard`: `max_kl=None` |
| `MultiAgentPPOTrainerConfig` | `MultiAgentPPOConfig` | as PPO |
| `MultiAgentA2CTrainerConfig` | `MultiAgentA2CConfig` | as A2C |

The loss settings are model fields at the same paths in every run, such as `model.clip_coef`; see [PPOModel](rl_models.md#ppomodel).

A plain `DragonRLConfig`, or another algorithm's model config, passed as `model` is converted. The settings both configs have are kept, and the rest take defaults. For example, a `PPOConfig` keeps `clip_coef` in `MultiAgentPPOTrainerConfig`, but not in `A2CTrainerConfig`.

### RolloutConfig

```python
class dragon_torch.trainers.RL.RolloutConfig(n_steps=128, gamma=0.99, gae_lambda=0.95, update_epochs=4,
                                             n_minibatches=4, normalize_obs=False, normalize_reward=False)
```

The settings for [`RolloutSource`](rl_models.md#rolloutsource). Each epoch collects `n_steps × n_envs` transitions.

### EvalConfig

```python
class dragon_torch.trainers.RL.EvalConfig(every_steps, n_episodes=10, smooth_window=3, deterministic=True,
                                          env=None, n_envs=1, seed_offset=10_000, max_steps=1_000_000)
```

**Parameters**
- **every_steps** (*int*) – evaluate every N env steps, and once more at `total_timesteps`. Keep it at least one epoch.
- **n_episodes** (*int*) – episodes per evaluation.
- **smooth_window** (*int*) – evaluations averaged into `val_return_smooth`.
- **deterministic** (*bool*) – greedy actions.
- **env** (*optional*) – a separate eval env. Default: the training env with `n_envs` sub-envs and seed `env.seed + seed_offset`, so evaluation never replays training episodes.
- **max_steps** (*int*) – vector-step cap per evaluation.

### PolicyConfig

```python
class dragon_torch.trainers.RL.PolicyConfig(hidden=(64, 64), activation='tanh')
```

The default network. `hidden` also accepts strings such as `'64x64'`. `activation` is one of `'tanh'`, `'relu'`, `'gelu'` or `'elu'`.

### OptimConfig

```python
class dragon_torch.trainers.RL.OptimConfig(name='adam', lr=3e-4, eps=1e-5, weight_decay=0.0, alpha=0.99,
                                           momentum=0.0, anneal_lr=False, final_lr_frac=0.0)
```

`name` is one of `'adam'`, `'adamw'`, `'rmsprop'` or `'sgd'`. With `anneal_lr`, the LR decays linearly to `lr × final_lr_frac` over `model.total_timesteps`.

### GuardConfig

```python
class dragon_torch.trainers.RL.GuardConfig(max_kl=0.5, kl_patience=2, min_entropy=None, entropy_after_steps=0)
```

The settings for [`DivergenceGuard`](#divergenceguard). `max_kl=None` disables the KL check.

### apply_overrides

```python
dragon_torch.trainers.RL.apply_overrides(cfg, params) -> config
```

Returns a copy of `cfg` with fields replaced by dotted path. Paths walk config sections, list indices and dict keys. Every changed section is re-validated, and unknown paths raise `KeyError`. Tuning applies sampled values this way.

```python
>>> apply_overrides(cfg, {'rollout.n_steps': 256, 'model.clip_coef': 0.1,
...                       'env.wrappers.0.kwargs.cost': 0.05})
```

Configs round-trip through JSON with `cfg.model_dump_json()` and `PPOTrainerConfig.model_validate_json(...)`.

## Callbacks

### StepEvalCallback

```python
class dragon_torch.trainers.RL.StepEvalCallback(evaluator, every_steps, total_steps=None,
                                                metric='val_return', smooth_window=3)
```

Evaluates every `every_steps` env steps. Trainers add it from `cfg.eval`. On the epochs where it evaluates, it adds these keys to the record:

| Key | Value |
|---|---|
| `val_return` | Mean return (raw, by default) |
| `val_return_smooth` | Mean of the last `smooth_window` evaluations; the default tuning objective |
| `val_return_raw`, `val_return_shaped` | Both returns, when the env has wrappers |
| `val_return/<group>` | Multi-agent runs: each group's mean return |
| `eval_step` | The scheduled step (a multiple of `every_steps`). It lines up across runs with different epoch sizes, unlike `global_step` |

### DivergenceGuard

```python
class dragon_torch.trainers.RL.DivergenceGuard(max_kl=0.5, patience=2, min_entropy=None,
                                               entropy_after_steps=0)
```

Raises `TrainingDiverged` when:

- `train_loss`, `policy_loss` or `value_loss` is NaN or inf,
- `approx_kl > max_kl` for `patience` epochs in a row, or
- `entropy < min_entropy` after `entropy_after_steps` env steps.

It checks only the keys the record has. In a tuning study, the exception becomes a pruned trial.

> **Note**
> `approx_kl` is measured before each minibatch's update, so with `update_epochs × n_minibatches == 1` (A2C) it is always 0 and the KL check can't fire.

## Notes

### LR annealing

`DragonTrainer(..., lr_schedule=fn)` sets each param group's LR to `initial_lr × fn(record)` after every epoch. `linear_schedule(total, key='epoch', final_frac=0.0)` is the linear decay that RL trainers use over `global_step`. The schedule is a function of the record, so a resumed run continues it. Every record includes the `lr` it trained with.

### Resuming

With `resume=True`, the first `fit()` loads the latest checkpoint in `model.checkpoint_dir`, restoring:

- the weights and optimizer state, including the annealed LR,
- the source's `global_step` and normalisers,
- the eval callback's smoothing window and schedule.

Training then continues to `total_timesteps`. Any object with `state_dict` / `load_state_dict` can join checkpoints through `model.register_state(name, obj)`.

### Adding an algorithm

Follow `models/RL/loss_configs.py`, `models/RL/*/ppo.py` and `trainers/RL/PPO/`:

```python
class MyLossConfig(BaseModel):                          # loss settings
    beta: float = 0.1
    def build_loss(self):
        return MyLoss(beta=self.beta)

class MyConfig(MyLossConfig, DragonRLConfig):           # model config
    kind: str = 'my-model'

class MyModel(SingleAgentRLModel):
    kind = 'my-model'
    def __init__(self, module, cfg: MyConfig, loss_fn=None):
        super().__init__(module, loss_fn or cfg.build_loss(), cfg)

class MyTrainerConfig(SingleAgentRLTrainerConfig):
    model: MyConfig
    @field_validator('model', mode='before')
    @classmethod
    def _model(cls, v):
        return upgrade_model_config(v, MyConfig)

class MyTrainer(SingleAgentRLTrainer):
    config_cls, model_cls = MyTrainerConfig, MyModel
```

For multi-agent support, add the same classes over `MultiAgentRLConfig`, `MultiAgentRLModel` (with the loss wrapped in `MultiAgentLoss`), `MultiAgentRLTrainerConfig` and `MultiAgentRLTrainer`. Register a tuning preset with `register_preset(MyModel, my_space)`.
