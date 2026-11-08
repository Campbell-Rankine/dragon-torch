# RL trainers

`dragon_torch/trainers/RL/` builds a complete on-policy RL run from a single config. That covers the rollout source, the evaluator, the policy network, the loss, the model, the optimizer, LR annealing, and the default callbacks.

```python
from dragon_torch.data.envs import DragonEnvConfig
from dragon_torch.models.RL import DragonRLConfig
from dragon_torch.trainers.RL import EvalConfig, OptimConfig, PPOTrainer, PPOTrainerConfig

cfg = PPOTrainerConfig(
    env=DragonEnvConfig(env_id='CartPole-v1', n_envs=4),
    model=DragonRLConfig(checkpoint_dir='runs/ppo', save_every=50, total_timesteps=25_000),
    eval=EvalConfig(every_steps=5_000, n_episodes=8),
    optim=OptimConfig(lr=2.5e-3, anneal_lr=True))

with PPOTrainer(cfg) as trainer:      # closes the envs on exit
    history = trainer.fit()           # trains on its own rollout source until total_timesteps
    score = trainer.evaluate()        # mean return on its own evaluator
```

Import from `dragon_torch.trainers.RL`. The plain `trainers` package doesn't import it, so supervised code doesn't need the `RL` extra.

## Layout

| File | Contents |
|---|---|
| `dragon_torch/trainers/RL/config.py` | `DragonRLTrainerConfig` and its sections, plus `apply_overrides` |
| `dragon_torch/trainers/RL/callbacks.py` | `StepEvalCallback`, `DivergenceGuard` and `EVAL_STEP_KEY` (`TrainingDiverged` itself is in `dragon_torch/trainers/base.py`) |
| `dragon_torch/trainers/RL/trainer.py` | `DragonRLTrainer`, the abstract base class |
| `dragon_torch/trainers/RL/PPO/` | `PPOLossConfig`, `PPOTrainerConfig`, `PPOTrainer` |
| `dragon_torch/trainers/RL/A2C/` | `A2CLossConfig`, `A2CTrainerConfig`, `A2CTrainer` |

## Config

`DragonRLTrainerConfig` extends `DragonTrainerConfig`, so it also has `epochs` and `resume`.

| Section | Type | Contents |
|---|---|---|
| `env` | `DragonEnvConfig` | The env to train on (required) |
| `model` | `DragonRLConfig` | `checkpoint_dir`, `save_every`, `total_timesteps`, `max_grad_norm`, `device` (required) |
| `rollout` | `RolloutConfig` | `n_steps`, `gamma`, `gae_lambda`, `update_epochs`, `n_minibatches`, `normalize_obs`, `normalize_reward` |
| `eval` | `EvalConfig \| None` | `every_steps`, `n_episodes`, `smooth_window`, `deterministic`, `env`, `n_envs`, `seed_offset`, `max_steps`. `None` means no evaluation |
| `policy` | `PolicyConfig` | `hidden` (a tuple, or a string such as `'64x64'`) and `activation` (`tanh`, `relu`, `gelu` or `elu`) |
| `optim` | `OptimConfig` | `name` (`adam`, `adamw`, `rmsprop` or `sgd`), `lr`, `eps`, `weight_decay`, `alpha`, `momentum`, `anneal_lr`, `final_lr_frac` |
| `guard` | `GuardConfig \| None` | `max_kl`, `kl_patience`, `min_entropy`, `entropy_after_steps`. `None` turns the guard off |
| `seed` | `int` | Torch seed for network initialisation |

**Algorithm sections**
- `PPOTrainerConfig.ppo`: `clip_coef`, `vf_coef`, `ent_coef`, `clip_vloss`, `normalize_advantages`. The rollout defaults to 4 update epochs of 4 minibatches.
- `A2CTrainerConfig.a2c`: `vf_coef`, `ent_coef`, `normalize_advantages`. The defaults follow the usual A2C recipe: 5-step rollouts, `gae_lambda=1`, one full-batch update per rollout, and RMSprop at a learning rate of 7e-4.

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

`DragonRLTrainer(cfg, module=None)` builds the run by calling these hooks in order. Override any of them:

1. `build_source` returns a `RolloutSource` built from `env` and `rollout`.
2. `build_evaluator` returns an `EpisodeSource`, or `None` when `eval` is `None`.
3. `build_module` returns an `ActorCritic` built from `policy`. It's skipped when you pass `module=`.
4. `build_loss` is **abstract**; it's where an algorithm is defined.
5. `build_model` returns a `DragonRLModel`.
6. `build_optimizer` returns `optim.factory()`.
7. `build_lr_schedule` returns a linear anneal over `global_step` when `optim.anneal_lr` is set.
8. `build_eval_callback` and `build_guard` build the default callbacks.

The trainer seeds torch from `cfg.seed` before building anything. If any step fails, it closes the envs it had already opened.

**Running**
- `fit(train_data=None, val_data=None, epochs=None, callbacks=())` trains on the trainer's own source until `model.total_timesteps`, unless you pass `epochs`.
- `default_callbacks()` (eval, then guard) run before your callbacks, so your callbacks see `val_return_smooth`.
- `evaluate()` uses the trainer's own evaluator.
- `close()` and `with` close the envs.

**Adding an algorithm**

A new algorithm needs only a config with a loss section and a `build_loss`:

```python
class MyAlgoTrainer(DragonRLTrainer):
    config: MyAlgoTrainerConfig
    def build_loss(self):
        return MyLoss(**self.config.my_algo.model_dump())
```

**Default tuning objective**

`objectives = {'val_return_smooth': 'maximize'}` is the default tuning objective, which `dragon_torch.utils.tuning.RL.rl_tune_config` reads. Objectives are a tuning choice, not a training one, so there are no separate single- and multi-objective trainers. The history contains every metric, and `TuneConfig` decides which ones to optimise.

## Evaluation

`StepEvalCallback` evaluates every `eval.every_steps` env steps, and once more at `total_timesteps`. Each evaluation adds three values to that epoch's record:

| Key | Value |
|---|---|
| `val_return` | Mean return over this evaluation's episodes |
| `val_return_smooth` | Mean of the last `smooth_window` evaluations |
| `eval_step` | The scheduled step: a multiple of `every_steps`, capped at `total_timesteps`. It is the same across runs with different epoch sizes, unlike `global_step` |

The eval env defaults to the training env with `eval.n_envs` sub-envs and `seed + seed_offset`, so evaluation episodes never replay training episodes. It also borrows the training source's observation normaliser.

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
- the rollout source's `global_step` and normaliser,
- the eval callback's smoothing window and schedule position. The trainer saves these through `DragonModel.register_state`, which checkpoints any object that has `state_dict` and `load_state_dict`.

The remaining `total_timesteps` budget then runs as normal.
