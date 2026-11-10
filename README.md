# dragon-torch

[![Tests on main](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FCampbell-Rankine%2Fdragon-torch%2Fbadges%2Ftests.json)](https://github.com/Campbell-Rankine/dragon-torch/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/dragon-torch)](https://pypi.org/project/dragon-torch/)

PyTorch Deep Learning Dev Modules

### Sources:

Define source for data in a dataset. Source.__getitem__() is cached and returns the full dataset. Supported types:

- disk
- hugging-face
- hugging-face-polars

### Datasets:

take a source as input. Dataset.__getitem__() returns an iterable of batches from the input source

**NOTE**: Dataset aggregators define batching rules / filters for combining two datasets

### Tests:

Pushes to `dev` run the full test suite in GitHub Actions; when every test passes, `dev` is merged into `main` automatically. Pushes to `main` run the tests without merging anything. The badge above shows how many tests pass on `main`, and is refreshed after every `main` run. Run them locally with:

```bash
uv run --extra test --extra torch-cpu --extra torch-ext pytest            # everything
uv run --extra test --extra torch-cpu --extra torch-ext pytest -m "not network"  # skip Hugging Face Hub tests
```

### Releases:

Install with `pip install dragon-torch`. Add extras as needed: `dragon-torch[RL,torch-cpu]` for RL training, `[tune]` for the dashboard. Everything imports from `dragon_torch`, for example `from dragon_torch.trainers.RL import PPOTrainer`.

When the tests pass on `main`, `tests.yml` starts `release.yml`. That workflow:

1. Re-checks that the commit is on `main` and that its `test` job passed.
2. Picks the next version with `scripts/release.py`.
3. Builds the package and uploads it to PyPI.
4. Tags the commit `vX.Y.Z`. A commit that already has a tag is never released twice.

Each release bumps the patch number of the latest release, carrying over when a part reaches its cap. The caps are 10, 10 and 20, so `1.1.20` is followed by `1.2.0`, `1.10.20` by `2.0.0`, and `10.10.20` is the last possible version. The version in `pyproject.toml` acts as a floor: set it higher, for example to `2.0.0`, to jump straight to that version.

Uploads use PyPI trusted publishing. Add a publisher for this repo on PyPI with workflow `release.yml` and environment `pypi`.

### Tuning / monitoring:

`dragon_torch/utils/tuning/` runs optuna studies over any `_DragonModel` (pruning, multi-objective, shared storage for parallel workers) and serves them in `optuna-dashboard`. `dragon_torch/utils/tuning/RL.py` and `dragon_torch/utils/tuning/presets/` tune any RL trainer, with one search-space preset per algorithm for its single- and multi-agent models (`space_for`); `scripts/tune_rl.py` runs a study worker. See [docs/tuning.md](docs/tuning.md).

```bash
uv sync --extra tune
dragon-tune dashboard --storage sqlite:///runs/optuna.db
```

### TODO:

- [ ] : Setup Lazy Loading for hugging face sources.
- [x] : Set up monitoring / experimentation with optuna
- [ ] : Implement off policy algorithms / dependencies.
- [ ] : Test Accelerate.prepare works with loaders
- [ ] : Test Accelerate.prepare works with gyms
- [ ] : OUActionNoise for RL 
- [ ] : Research better state memory solutions
- [ ] : Trading environment as a test case in self implemented problem


### RL

| Layer | Files | Contents |
|---|---|---|
| Env utils | `dragon_torch/data/envs/` | `DragonEnvConfig` and `make_vec_env`, which always uses `SAME_STEP` autoreset. Wrappers can be given as data (`WrapperSpec`, `register_wrapper`; see [docs/envs.md](docs/envs.md)). Also `validate_env`, `register_dragon_env`, space conversions (`obs_to_tensor`, `action_to_env`), running normalisation, and `Discretizer` for stable-retro's button actions |
| Sources | `dragon_torch/data/RL/` | `EnvCollector` (env stepping, shared by all sources), the `Actor` protocol, `Transition`, `RolloutSource` with GAE, and `EpisodeSource` for evaluation. Multi-agent versions of each are in `multi_agent.py` |
| Multi-agent envs | `dragon_torch/data/envs/multi_agent.py` | `MultiAgentEnvConfig` for PettingZoo-style `ParallelEnv`s and gymnasium envs with per-agent `Dict` spaces, agent groups (`AgentGroups`) and `MultiAgentVectorEnv`. See [docs/multi_agent.md](docs/multi_agent.md) |
| Networks | `dragon_torch/modules/RL/` | `SingleAgentPolicyModule` / `MultiAgentPolicyModule`; `ActorCritic` and `MultiAgentActorCritic` (one `ActorCritic` per agent group); `PolicyOutput`; and distribution heads for Discrete, MultiDiscrete, MultiBinary and Box actions |
| Losses | `dragon_torch/loss/RL.py` | `PPOLoss` and `A2CLoss`, which each record their components (KL, clip fraction, entropy), and `MultiAgentLoss`, which applies either one to each agent group |
| Model | `dragon_torch/models/RL/` | `DragonRLModel` (abstract) adds gradient clipping and returns episode return from `evaluate`. Its epoch count comes from `total_timesteps`, minus the steps already taken. `single_agent/` has `SingleAgentRLModel` and the algorithm models `PPOModel` / `A2CModel`; `multi_agent/` has `MultiAgentRLModel`, `MultiAgentPPOModel` and `MultiAgentA2CModel`. Each algorithm model's config holds its loss settings (`model.clip_coef`, ...) |
| Trainers | `dragon_torch/trainers/RL/` | `DragonRLTrainer` builds a whole run from one config: rollout source, evaluator, network, loss, optimizer, LR annealing, step-scheduled eval and a divergence guard. `SingleAgentRLTrainer` and `MultiAgentRLTrainer` handle the env kind, and algorithm mixins supply the loss: `PPOTrainer`, `A2CTrainer`, `MultiAgentPPOTrainer` and `MultiAgentA2CTrainer`. See [docs/rl_trainers.md](docs/rl_trainers.md) |
| SB3 | `dragon_torch/modules/external/sb3.py` | `DragonFeaturesExtractor` and `sb3_policy_kwargs` |


