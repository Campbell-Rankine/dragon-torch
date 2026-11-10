# dragon-torch

[![Tests on main](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FCampbell-Rankine%2Fdragon-torch%2Fbadges%2Ftests.json)](https://github.com/Campbell-Rankine/dragon-torch/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/dragon-torch)](https://pypi.org/project/dragon-torch/)

A PyTorch research framework focused on reinforcement learning. It provides environments, policies, losses, models and trainers that compose, plus Optuna-based hyperparameter tuning.

## Installation

```bash
pip install "dragon-torch[RL,torch-cpu]"
```

| Extra | Adds |
|---|---|
| `RL` | Gymnasium, PettingZoo, stable-retro, Stable-Baselines3 |
| `tune` | Optuna and optuna-dashboard |
| `torch-cpu`, `torch-cuda12`, `torch-cuda13` | PyTorch builds |
| `torch-ext` | Transformers, Accelerate, Datasets |
| `test` | pytest |

## Quick start

```python
from dragon_torch.data.envs import DragonEnvConfig
from dragon_torch.models.RL import PPOConfig
from dragon_torch.trainers.RL import EvalConfig, PPOTrainer, PPOTrainerConfig

cfg = PPOTrainerConfig(
    env=DragonEnvConfig(env_id='CartPole-v1', n_envs=4),
    model=PPOConfig(checkpoint_dir='runs/ppo', save_every=50, total_timesteps=25_000),
    eval=EvalConfig(every_steps=5_000))

with PPOTrainer(cfg) as trainer:
    history = trainer.fit()
```

New to dragon-torch? The [tutorial](docs/tutorial.md) walks through this example step by step.

## Documentation

See [docs/index.md](docs/index.md).

| Package | Contents |
|---|---|
| [`dragon_torch.data.envs`](docs/envs.md) | Env configs, wrappers as data, env checks |
| [`dragon_torch.models.RL`, `.modules.RL`, `.loss.RL`, `.data.RL`](docs/rl_models.md) | `PPOModel` / `A2CModel`, `ActorCritic`, losses, rollout sources |
| [`dragon_torch.trainers.RL`](docs/rl_trainers.md) | `PPOTrainer`, `A2CTrainer` and their multi-agent versions |
| [`dragon_torch.utils.tuning`](docs/tuning.md) | Optuna studies, search-space presets, the `dragon-tune` CLI |
| [Multi-agent RL](docs/multi_agent.md) | PettingZoo `ParallelEnv`s and Gymnasium `Dict`-space envs, agent groups |
| `dragon_torch.data.sources`, `.datasets` | Data sources (disk, Hugging Face, Hugging Face + Polars) and the datasets that batch them. `Source.__getitem__` is cached and returns the full dataset; aggregators combine datasets |
| `dragon_torch.modules.external.sb3` | `DragonFeaturesExtractor` and `sb3_policy_kwargs`, for using dragon modules in Stable-Baselines3 |

## Development

**Tests.** Pushes to `dev` run the test suite in GitHub Actions and, when it passes, merge `dev` into `main`. Pushes to `main` run the tests without merging. The badge shows the pass count on `main`. To run the tests locally:

```bash
uv run --extra test --extra torch-cpu --extra torch-ext pytest                       # everything
uv run --extra test --extra torch-cpu --extra torch-ext pytest -m "not network"      # skip Hugging Face Hub tests
```

**Releases.** A passing `main` run starts `release.yml`, which:

1. re-checks that the commit is on `main` and that its tests passed,
2. picks the next version with `scripts/release.py`,
3. builds the package and uploads it to PyPI through trusted publishing (workflow `release.yml`, environment `pypi`),
4. tags the commit `vX.Y.Z`. A tagged commit is never released twice.

Versions bump the patch number. Parts carry over at caps of 10, 10 and 20, so `1.1.20` is followed by `1.2.0`, and `10.10.20` is the last version. The version in `pyproject.toml` is a floor: raise it to jump straight to that version.

## TODO

- [ ] Lazy loading for Hugging Face sources
- [x] Monitoring and experimentation with Optuna
- [ ] Off-policy algorithms
- [ ] Test `Accelerate.prepare` with loaders and gyms
- [ ] `OUActionNoise` for RL
- [ ] Better state memory solutions
- [ ] A trading environment as a self-implemented test problem
