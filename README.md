# dragon-torch

[![Tests on main](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FCampbell-Rankine%2Fdragon-torch%2Fbadges%2Ftests.json)](https://github.com/Campbell-Rankine/dragon-torch/actions/workflows/tests.yml)

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

### Tuning / monitoring:

`utils/tuning/` runs optuna studies over any `_DragonModel` (pruning, multi-objective, shared storage for parallel workers) and serves them in `optuna-dashboard`. See [docs/tuning.md](docs/tuning.md).

```bash
uv sync --extra tune
python -m utils.tuning.monitor dashboard --storage sqlite:///runs/optuna.db
```

### TODO:

- [ ] : Setup Lazy Loading for hugging face sources.
- [ ] : Set up monitoring / experimentation with optuna
- [ ] : Implement off policy algorithms / dependencies.
- [ ] : Test Accelerate.prepare works with loaders
- [ ] : Test Accelerate.prepare works with gyms
- [ ] : OUActionNoise for RL 
- [ ] : Research better state memory solutions
- [ ] : Trading environment as a test case in self implemented problem


### RL

| Layer | Files | Contents |
|---|---|---|
| Env utils | `data/envs/` | `DragonEnvConfig` and `make_vec_env`, which always uses `SAME_STEP` autoreset. Also `validate_env`, `register_dragon_env`, space conversions (`obs_to_tensor`, `action_to_env`), running normalisation, and `Discretizer` for stable-retro's button actions |
| Sources | `data/RL/` | `EnvCollector` (env stepping, shared by all sources), the `Actor` protocol, `Transition`, `RolloutSource` with GAE, and `EpisodeSource` for evaluation |
| Networks | `modules/RL/` | `ActorCritic`, `PolicyOutput`, and distribution heads for Discrete, MultiDiscrete, MultiBinary and Box actions |
| Losses | `loss/RL.py` | `PPOLoss` and `A2CLoss`; each records its components (KL, clip fraction, entropy) |
| Model | `models/RL/base.py` | `DragonRLModel` adds gradient clipping and returns episode return from `evaluate`. Its epoch count comes from `total_timesteps`, minus the steps already taken |
| SB3 | `modules/external/sb3.py` | `DragonFeaturesExtractor` and `sb3_policy_kwargs` |


