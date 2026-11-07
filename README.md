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

### TODO:

- [ ] : Setup Lazy Loading for hugging face sources.


### RL

| Layer | Files | Contents |
|---|---|---|
| Env utils | `data/envs/` | `DragonEnvConfig` and `make_vec_env`, which always uses `SAME_STEP` autoreset. Also `validate_env`, `register_dragon_env`, space conversions (`obs_to_tensor`, `action_to_env`), running normalisation, and `Discretizer` for stable-retro's button actions |
| Sources | `data/RL/` | `EnvCollector` (env stepping, shared by all sources), the `Actor` protocol, `Transition`, `RolloutSource` with GAE, and `EpisodeSource` for evaluation |
| Networks | `modules/RL/` | `ActorCritic`, `PolicyOutput`, and distribution heads for Discrete, MultiDiscrete, MultiBinary and Box actions |
| Losses | `loss/RL.py` | `PPOLoss` and `A2CLoss`; each records its components (KL, clip fraction, entropy) |
| Model | `models/RL/base.py` | `DragonRLModel` adds gradient clipping and returns episode return from `evaluate`. Its epoch count comes from `total_timesteps`, minus the steps already taken |
| SB3 | `modules/external/sb3.py` | `DragonFeaturesExtractor` and `sb3_policy_kwargs` |

`DragonTrainer` drives RL models unchanged. `models/base.py` has two hooks for them: `epoch_metrics()`, so loss components reach the history, and `eval_metric`, so RL records `val_return` instead of `val_loss`.

#### Behaviours worth knowing

- **Truncated episodes bootstrap correctly.**
  - When an episode ends, the collector puts the true final observation into the transition. The policy still sees the reset observation on its next step.
  - Truncated episodes bootstrap from the final observation's value; terminated ones don't bootstrap at all.
  - Tests check both cases against hand-computed advantages.
- **Normalisation lives in the collector, not in gym wrappers.**
  - The same statistics also apply to the final observations, which wrappers wouldn't reach.
  - Evaluation borrows the training statistics read-only.
  - The statistics are stored as tensors, so this source state would also load under torch's safe loader.
- **Resuming works.** A test checks that resume restores the normalisation stats and the step budget, then finishes the remaining epochs.
