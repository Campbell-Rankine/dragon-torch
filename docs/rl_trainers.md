# RL trainers

`dragon_torch/trainers/RL/` builds a complete on-policy RL run from a single config. That covers the rollout source, the evaluator, the policy network, the loss, the model, the optimizer, LR annealing, and the default callbacks.

```python
from dragon_torch.data.envs import DragonEnvConfig
from dragon_torch.models.RL import PPOConfig
from dragon_torch.trainers.RL import EvalConfig, OptimConfig, PPOTrainer, PPOTrainerConfig

cfg = PPOTrainerConfig(
    env=DragonEnvConfig(env_id='CartPole-v1', n_envs=4),
    model=PPOConfig(checkpoint_dir='runs/ppo', save_every=50, total_timesteps=25_000,
                    ent_coef=0.01),               # PPO's loss settings live on its model config
    eval=EvalConfig(every_steps=5_000, n_episodes=8),
    optim=OptimConfig(lr=2.5e-3, anneal_lr=True))

with PPOTrainer(cfg) as trainer:      # closes the envs on exit
    history = trainer.fit()           # trains on its own rollout source until total_timesteps
    score = trainer.evaluate()        # mean return on its own evaluator
```

Import from `dragon_torch.trainers.RL`. The plain `trainers` package doesn't import it, so supervised code doesn't need the `RL` extra.

Every algorithm comes in a single-agent and a multi-agent version: `PPOTrainer` / `MultiAgentPPOTrainer` and `A2CTrainer` / `MultiAgentA2CTrainer`. Each trainer trains an algorithm model class: `PPOModel` / `MultiAgentPPOModel` or `A2CModel` / `MultiAgentA2CModel` (see [Algorithm models](#algorithm-models)). This page covers what they share. [multi_agent.md](multi_agent.md) covers multi-agent envs, agent groups and the multi-agent metrics.

## Layout

| File | Contents |
|---|---|
| `dragon_torch/trainers/RL/config.py` | `DragonRLTrainerConfig` (shared sections), `SingleAgentRLTrainerConfig`, `MultiAgentRLTrainerConfig`, plus `apply_overrides` |
| `dragon_torch/trainers/RL/callbacks.py` | `StepEvalCallback`, `DivergenceGuard` and `EVAL_STEP_KEY` (`TrainingDiverged` itself is in `dragon_torch/trainers/base.py`) |
| `dragon_torch/trainers/RL/trainer.py` | `DragonRLTrainer` (abstract base), `SingleAgentRLTrainer`, `MultiAgentRLTrainer` |
| `dragon_torch/trainers/RL/PPO/` | `PPODefaults`, `PPOTrainer(Config)`, `MultiAgentPPOTrainer(Config)`, plus re-exports of the PPO models and configs |
| `dragon_torch/trainers/RL/A2C/` | `A2CDefaults`, `A2CTrainer(Config)`, `MultiAgentA2CTrainer(Config)`, plus re-exports of the A2C models and configs |
| `dragon_torch/models/RL/loss_configs.py` | `PPOLossConfig`, `A2CLossConfig`: each algorithm's loss settings, shared by its model configs |
| `dragon_torch/models/RL/single_agent/` | `SingleAgentRLModel`, `PPOModel` / `PPOConfig`, `A2CModel` / `A2CConfig` |
| `dragon_torch/models/RL/multi_agent/` | `MultiAgentRLModel`, `MultiAgentPPOModel` / `MultiAgentPPOConfig`, `MultiAgentA2CModel` / `MultiAgentA2CConfig` |

## Config

`DragonRLTrainerConfig` holds the sections every run shares. It extends `DragonTrainerConfig`, so it also has `epochs` and `resume`. Two subclasses fix the env type, and the algorithm configs build on them:

- `SingleAgentRLTrainerConfig` takes a `DragonEnvConfig` (gymnasium). `PPOTrainerConfig` and `A2CTrainerConfig` build on it.
- `MultiAgentRLTrainerConfig` takes a `MultiAgentEnvConfig`. `MultiAgentPPOTrainerConfig` and `MultiAgentA2CTrainerConfig` build on it.

| Section | Type | Contents |
|---|---|---|
| `env` | `DragonEnvConfig` / `MultiAgentEnvConfig` | The env to train on (required) |
| `model` | `DragonRLConfig` | `checkpoint_dir`, `save_every`, `total_timesteps`, `max_grad_norm`, `device` (required). Each algorithm config narrows it to the algorithm's model config, which holds the loss settings (see below). A plain `DragonRLConfig` is converted, and so is another model config: the settings the two have in common are kept, and the rest take defaults. For example, a `PPOConfig` given to `MultiAgentPPOTrainerConfig` keeps `clip_coef`, but given to `A2CTrainerConfig` it keeps only the shared `DragonRLConfig` settings |
| `rollout` | `RolloutConfig` | `n_steps`, `gamma`, `gae_lambda`, `update_epochs`, `n_minibatches`, `normalize_obs`, `normalize_reward` |
| `eval` | `EvalConfig \| None` | `every_steps`, `n_episodes`, `smooth_window`, `deterministic`, `env`, `n_envs`, `seed_offset`, `max_steps`. `None` means no evaluation |
| `policy` | `PolicyConfig` | `hidden` (a tuple, or a string such as `'64x64'`) and `activation` (`tanh`, `relu`, `gelu` or `elu`) |
| `optim` | `OptimConfig` | `name` (`adam`, `adamw`, `rmsprop` or `sgd`), `lr`, `eps`, `weight_decay`, `alpha`, `momentum`, `anneal_lr`, `final_lr_frac` |
| `guard` | `GuardConfig \| None` | `max_kl`, `kl_patience`, `min_entropy`, `entropy_after_steps`. `None` turns the guard off |
| `seed` | `int` | Torch seed for network initialisation |
| `validate_env` | `bool` | Check the wrapped env before building anything (default `False`). Single-agent runs use Gymnasium's env checker (see [envs.md](envs.md#checking-an-env-before-training)); multi-agent runs use a short random rollout (see [multi_agent.md](multi_agent.md#checking-an-env)) |

**Algorithm settings**

An algorithm's loss settings are fields of its model config, at the same paths (`model.clip_coef`, ...) in single- and multi-agent runs:

| Algorithm | Loss settings | `model` type: single-agent / multi-agent |
|---|---|---|
| PPO | `clip_coef`, `vf_coef`, `ent_coef`, `clip_vloss`, `normalize_advantages` | `PPOConfig` / `MultiAgentPPOConfig` |
| A2C | `vf_coef`, `ent_coef`, `normalize_advantages` | `A2CConfig` / `MultiAgentA2CConfig` |

Each algorithm's defaults (`PPODefaults`, `A2CDefaults`) are shared by both kinds of run:

- **PPO:** the rollout defaults to 4 update epochs of 4 minibatches.
- **A2C:** the usual A2C recipe. That means 5-step rollouts, `gae_lambda=1`, one full-batch update per rollout, RMSprop at a learning rate of 7e-4, and a guard without the KL check.

**Validation**
- `rollout.n_steps × env.n_envs` must be divisible by `rollout.n_minibatches`. Otherwise `RolloutSource` would silently drop the leftover samples.
- `optim.anneal_lr` requires `model.total_timesteps`.
- If `eval.every_steps` is smaller than one epoch, the config warns, because every epoch would evaluate and scheduled steps would be skipped.
- If `guard.max_kl` is set but `update_epochs × n_minibatches == 1`, the config warns, because `approx_kl` is then always 0 (see [Divergence guard](#divergence-guard)).

**`apply_overrides`**

`apply_overrides(cfg, {'rollout.n_steps': 256, 'optim.lr': 3e-4})` returns a copy with those fields replaced by dotted path. Paths can also index into lists and dicts, for example `'env.wrappers.0.kwargs.cost'` (see [envs.md](envs.md#tuning-wrapper-settings)). Every section it changes is re-validated, so the checks above still run, and unknown keys raise an error. Tuning search spaces use this to apply sampled values.

**JSON**

Configs round-trip through JSON with `model_dump_json()` and `PPOTrainerConfig.model_validate_json(...)`.

## Trainer

A concrete trainer combines two choices:

- **The kind of env.** `SingleAgentRLTrainer` and `MultiAgentRLTrainer` implement the env-specific hooks.
- **The algorithm.** `model_cls` names the algorithm's model class, for example `PPOTrainer.model_cls = PPOModel` and `MultiAgentPPOTrainer.model_cls = MultiAgentPPOModel`. `build_agent_loss` builds the single-agent loss from the model config with `config.model.build_loss()`. Override it for a loss that isn't set by the config.

A trainer only accepts its own config type (`config_cls`) and raises `TypeError` for any other.

`DragonRLTrainer(cfg, module=None)` builds the run by calling these hooks in order. Override any of them. Before the first, `check_envs` checks the env when `validate_env` is set.

| Hook | Single-agent | Multi-agent |
|---|---|---|
| 1. `build_source` | `RolloutSource` | `MultiAgentRolloutSource` |
| 2. `build_evaluator` (`None` when `eval` is `None`) | `EpisodeSource` | `MultiAgentEpisodeSource` |
| 3. `build_module` (skipped when you pass `module=`) | `ActorCritic` from `policy` | `MultiAgentActorCritic`: one `ActorCritic` per agent group |
| 4. `build_loss` | `build_agent_loss()` | `MultiAgentLoss(build_agent_loss())` |
| 5. `build_model` | `model_cls(module=..., loss_fn=..., cfg=config.model)`, e.g. `PPOModel` | the same, e.g. `MultiAgentPPOModel` |

The remaining hooks are shared:

1. `build_optimizer` returns `optim.factory()`.
2. `build_lr_schedule` returns a linear anneal over `global_step` when `optim.anneal_lr` is set.
3. `build_eval_callback` and `build_guard` build the default callbacks.

The trainer seeds torch from `cfg.seed` before building anything. If any step fails, it closes the envs it had already opened.

**Running**
- `fit(train_data=None, val_data=None, epochs=None, callbacks=())` trains on the trainer's own source until `model.total_timesteps`, unless you pass `epochs`.
- `default_callbacks()` (eval, then guard) run before your callbacks, so your callbacks see `val_return_smooth`.
- `evaluate()` uses the trainer's own evaluator.
- `close()` and `with` close the envs.

**Adding an algorithm**

A new algorithm needs a single-agent `RLLoss` and the pieces below. Follow `models/RL/loss_configs.py`, `models/RL/*/ppo.py` and `trainers/RL/PPO/`:

```python
class MyLossConfig(BaseModel):                                 # the loss settings
    beta: float = 0.1
    def build_loss(self):
        return MyLoss(**self.model_dump(include=set(MyLossConfig.model_fields)))

class MyConfig(MyLossConfig, DragonRLConfig):                  # single agent
    kind: str = 'my-model'

class MyModel(SingleAgentRLModel):
    kind = 'my-model'
    def __init__(self, module, cfg: MyConfig, loss_fn=None):
        super().__init__(module, loss_fn if loss_fn is not None else cfg.build_loss(), cfg)

class MultiAgentMyConfig(MyLossConfig, MultiAgentRLConfig):    # multi agent
    kind: str = 'marl-my-model'

class MultiAgentMyModel(MultiAgentRLModel):
    kind = 'marl-my-model'
    def __init__(self, module, cfg: MultiAgentMyConfig, loss_fn=None):
        super().__init__(module, loss_fn if loss_fn is not None
                         else MultiAgentLoss(cfg.build_loss()), cfg)

class MyTrainerConfig(SingleAgentRLTrainerConfig):
    model: MyConfig

    @field_validator('model', mode='before')
    @classmethod
    def _my_model(cls, v):          # accept a plain DragonRLConfig too
        return upgrade_model_config(v, MyConfig)

class MyTrainer(SingleAgentRLTrainer):
    config_cls, model_cls = MyTrainerConfig, MyModel

# MultiAgentMyTrainerConfig / MultiAgentMyTrainer: the same, over
# MultiAgentRLTrainerConfig / MultiAgentRLTrainer with MultiAgentMyConfig / MultiAgentMyModel
```

Shared defaults go in a mixin config listed first among the bases (`class MyTrainerConfig(MyDefaults, SingleAgentRLTrainerConfig)`), so they take precedence over the base config's. Register a tuning preset for both models with `register_preset(MyModel, my_space)` and `register_preset(MultiAgentMyModel, my_space)` (see [tuning.md](tuning.md#search-spaces)).

**Default tuning objective**

`objectives = {'val_return_smooth': 'maximize'}` is the default tuning objective, which `dragon_torch.utils.tuning.RL.rl_tune_config` reads. Objectives are a tuning choice, not a training one, so there are no separate single- and multi-objective trainers. The history contains every metric, and `TuneConfig` decides which ones to optimise.

## Algorithm models

The algorithm models (in `dragon_torch.models.RL`) build their own loss from their config. They work without a trainer:

| Algorithm | Single-agent (`SingleAgentRLModel`) | Multi-agent (`MultiAgentRLModel`) |
|---|---|---|
| PPO | `PPOModel(module, PPOConfig(...))` | `MultiAgentPPOModel(module, MultiAgentPPOConfig(...))` |
| A2C | `A2CModel(module, A2CConfig(...))` | `MultiAgentA2CModel(module, MultiAgentA2CConfig(...))` |

```python
from dragon_torch.models.RL import PPOConfig, PPOModel
from dragon_torch.modules.RL import ActorCritic

model = PPOModel(ActorCritic(obs_space, act_space),
                 PPOConfig(checkpoint_dir='runs/ppo', save_every=10, clip_coef=0.1))
model.fit(RolloutSource(env_cfg), optimizer)
```

- **Configs.** Each config is the base model config (`DragonRLConfig` or `MultiAgentRLConfig`) plus the algorithm's loss settings (`PPOLossConfig`, `A2CLossConfig`).
- **Kinds.** The configs set `kind`: `'ppo-model'`, `'a2c-model'`, `'marl-ppo-model'` and `'marl-a2c-model'`. `model_factory` therefore builds the matching class, and checkpoints are named after it.
- **Losses.** The single-agent models train on the algorithm's loss. The multi-agent models wrap it in a `MultiAgentLoss`, which applies it to each agent group. Passing `loss_fn=` replaces the loss the config would build.
- **Validation.** A model rejects any other config type with `TypeError`.

## Evaluation

`StepEvalCallback` evaluates every `eval.every_steps` env steps, and once more at `total_timesteps`. Each evaluation adds three values to that epoch's record:

| Key | Value |
|---|---|
| `val_return` | Mean return over this evaluation's episodes: the env's raw reward by default (`model.eval_return`) |
| `val_return_raw`, `val_return_shaped` | Both returns, when `env.wrappers` is non-empty (see [envs.md](envs.md#raw-and-shaped-rewards)) |
| `val_return_smooth` | Mean of the last `smooth_window` evaluations |
| `val_return/<group>` | Multi-agent only: each agent group's mean per-agent return |
| `eval_step` | The scheduled step: a multiple of `every_steps`, capped at `total_timesteps`. It is the same across runs with different epoch sizes, unlike `global_step` |

The eval env defaults to the training env with `eval.n_envs` sub-envs and `seed + seed_offset`, so evaluation episodes never replay training episodes. It also borrows the training source's observation normaliser, frozen, which is why normalisation belongs in `rollout.normalize_obs` rather than in a wrapper (see [envs.md](envs.md#normalisation)).

## LR annealing

`DragonTrainer(..., lr_schedule=fn)` takes a function from an epoch record to a multiplier (`record -> multiplier`). After each epoch, it sets every param group's LR to `initial_lr × multiplier` for the next epoch. Every record also gets the `lr` it trained with, whether or not a schedule is set.

`linear_schedule(total, key, final_frac)` is the standard linear decay. RL trainers use it over `global_step` when `optim.anneal_lr` is set.

The schedule is a function of the record rather than a stateful torch scheduler. A resumed run therefore continues from where it stopped: the current LR comes back with the optimizer checkpoint, and `initial_lr` is stored in the param groups.

## Divergence guard

`DivergenceGuard` raises `dragon_torch.trainers.TrainingDiverged` on any of:

- a NaN or inf `train_loss`, `policy_loss` or `value_loss`
- `approx_kl > max_kl` for `kl_patience` epochs in a row
- `entropy < min_entropy` after `entropy_after_steps` env steps

The guard only checks keys the record contains, so A2C, which records no KL, works too. Outside tuning, the exception stops `fit`. Inside a study, the objective turns it into a pruned trial.

**The KL check needs more than one update per rollout.** `PPOLoss` measures `approx_kl` before each minibatch's optimizer step, so the first minibatch of every rollout always reads 0. When `rollout.update_epochs × rollout.n_minibatches == 1`, every recorded `approx_kl` is 0 and the KL check can't fire:

- The config warns if `guard.max_kl` is set in that case.
- `A2CTrainerConfig` defaults to `GuardConfig(max_kl=None)`, which keeps the NaN/inf check.

## Checkpoints and resuming

With `resume=True`, the first `fit()` loads the latest checkpoint in `model.checkpoint_dir` and restores:

- the weights and optimizer state, including the annealed LR,
- the rollout source's `global_step` and normalisers,
- the eval callback's smoothing window and schedule position. The trainer saves these through `DragonModel.register_state`, which checkpoints any object that has `state_dict` and `load_state_dict`.

The remaining `total_timesteps` budget then runs as normal.
