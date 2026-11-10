# dragon-torch documentation

dragon-torch is a PyTorch research framework built around reinforcement learning: environments, policies, losses, models and trainers that compose, plus Optuna-based hyperparameter tuning.

## Tutorials

| Page | Contents |
|---|---|
| [Train an agent to balance a pole](tutorial.md) | Start here: a PPO agent on CartPole, step by step, in about 10 seconds |

## Package reference

| Page | Package | Contents |
|---|---|---|
| [Environments](envs.md) | `dragon_torch.data.envs` | Env configs, wrappers as data, vectorisation, env checks |
| [RL components](rl_models.md) | `dragon_torch.models.RL`, `.modules.RL`, `.loss.RL`, `.data.RL` | Models (`PPOModel`, `A2CModel`, ...), policy networks, losses, rollout sources |
| [RL trainers](rl_trainers.md) | `dragon_torch.trainers.RL` | One-config training runs: `PPOTrainer`, `A2CTrainer` and their multi-agent versions |
| [Tuning](tuning.md) | `dragon_torch.utils.tuning` | Optuna studies, search spaces, presets, the `dragon-tune` CLI |

## Notes

| Page | Contents |
|---|---|
| [Multi-agent RL](multi_agent.md) | Multi-agent envs, agent groups, batching and metrics |
| [Raw and shaped rewards](envs.md#raw-and-shaped-rewards) | What `val_return` measures when wrappers shape the reward |
| [Normalisation](envs.md#normalisation) | Why normalisation lives in the rollout config, not in wrappers |

## Conventions

- Signatures list the defaults. Config classes are [pydantic](https://docs.pydantic.dev) models, so they validate on construction and round-trip through JSON.
- Configs are nested and addressed by dotted path, for example `rollout.n_steps` or `model.clip_coef`. Those paths are used by `apply_overrides` and by tuning search spaces.
- RL needs the `RL` extra, and tuning needs the `tune` extra.
