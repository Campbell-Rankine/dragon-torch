# Tutorial: train an agent to balance a pole

In this tutorial you'll train a PPO agent on CartPole, then evaluate it, use it, and reload it from a checkpoint. It takes about 10 seconds on a laptop CPU.

**You'll learn how to**
- describe an environment with `DragonEnvConfig`,
- configure an algorithm with its model config, here `PPOConfig`,
- build and run a training run with `PPOTrainer`,
- read the training history,
- use and reload a trained model.

**Prerequisites:** `pip install "dragon-torch[RL,torch-cpu]"`.

## The task

In [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/), a pole stands on a cart, and each step the agent pushes the cart left or right. It earns +1 for every step the pole stays up, and episodes end when the pole falls or after 500 steps. A random agent scores about 20, and a perfect one scores 500.

We'll use **PPO** (Proximal Policy Optimisation). It alternates between two phases: collecting a batch of experience with the current policy, and taking a few careful gradient steps on that batch. It's a robust default for problems like this one.

## 1. Describe the environment

```python
from dragon_torch.data.envs import DragonEnvConfig

env = DragonEnvConfig(env_id='CartPole-v1', n_envs=8, seed=0)
```

**What:** a config naming a registered Gymnasium env, run as 8 copies.

**Why:**
- **A config, not an env object.** It can be saved, validated and rebuilt, and the trainer builds the envs it needs from it, including a separate copy for evaluation.
- **8 copies (`n_envs=8`).** They're stepped together, so each batch mixes experience from 8 independent episodes. That gives steadier gradients, and the policy network runs on all 8 observations at once.
- **`seed=0`.** Makes the run reproducible.

## 2. Configure the algorithm

```python
from dragon_torch.models.RL import PPOConfig

model = PPOConfig(checkpoint_dir='runs/cartpole', save_every=10, total_timesteps=100_000)
```

**What:** the settings of the model we're training: a `PPOModel`.

**Why:**
- **`total_timesteps`** is the training budget, in env steps summed over all 8 copies. 100,000 is plenty for CartPole.
- **`checkpoint_dir` and `save_every`** save the model every 10 epochs and at the end, so it can be reloaded later (step 6).
- **PPO's own settings**, such as `clip_coef` (how far one update may move the policy) and `ent_coef` (a bonus for exploring), are also fields here. We keep the defaults. To change one, pass it in, for example `PPOConfig(..., ent_coef=0.01)`.

## 3. Decide how to measure progress

```python
from dragon_torch.trainers.RL import EvalConfig

evaluation = EvalConfig(every_steps=10_000, n_episodes=10)
```

**What:** every 10,000 env steps, play 10 episodes and record the mean return.

**Why:**
- **A separate evaluation.** During training, actions are *sampled* from the policy so the agent keeps exploring, which makes training returns understate the policy. Evaluation picks each action greedily on a separate env with a different seed, which measures what the agent has actually learned.
- **Scheduled by env steps rather than epochs.** Runs with different settings are then evaluated at the same points, which is what tuning compares (see [Tuning](tuning.md)).

## 4. Build the run and train

```python
from dragon_torch.trainers.RL import PPOTrainer, PPOTrainerConfig

cfg = PPOTrainerConfig(env=env, model=model, eval=evaluation)

with PPOTrainer(cfg) as trainer:
    history = trainer.fit()
```

**What:** `PPOTrainerConfig` combines the pieces into one run, and `PPOTrainer` builds and trains it.

**Why:**
- **One config.** It describes the whole run, and the sections we didn't set (`rollout`, `optim`, `policy`, `guard`) take PPO's defaults. The config is validated as it's built: for example, it checks that each batch splits evenly into minibatches.
- **The trainer builds everything from the config.** That includes the envs, a policy network sized to CartPole's observation and action spaces, the PPO loss, an Adam optimizer, the evaluation schedule, and a guard that stops training if it diverges. `with` closes the envs afterwards.
- **What each epoch does.** `fit()` runs epochs until `total_timesteps` is reached. Each epoch:
  1. collects 128 steps from each of the 8 envs (1,024 transitions),
  2. computes advantages (how much better each action was than expected),
  3. makes 4 passes over the batch in 4 minibatches.

## 5. Read the results

`fit()` returns one record per epoch. The epochs that ran an evaluation have `val_return`:

```python
for record in history:
    if 'val_return' in record:
        print(f"step {record['eval_step']:>7,}  return {record['val_return']:5.1f}  "
              f"smoothed {record['val_return_smooth']:5.1f}")
```

```
step  10,000  return 186.4  smoothed 186.4
step  20,000  return 275.9  smoothed 231.1
step  30,000  return 500.0  smoothed 320.8
step  40,000  return 252.1  smoothed 342.7
...
step 100,000  return 500.0  smoothed 377.8
```

The agent goes from about 20 to the maximum of 500. Your numbers will differ, because RL training is noisy: returns can jump to 500 and fall back before settling. That's why there's `val_return_smooth`, the mean of the last 3 evaluations. It's the better guide to progress, and it's what tuning optimises.

Each record also holds training diagnostics:
- `episode_return` – returns of the training episodes,
- `policy_loss`, `value_loss` and `entropy` – the loss components (entropy falls as the policy grows confident),
- `approx_kl` – how far each update moved the policy,
- `lr` and `steps_per_sec`.

## 6. Use and reload the trained model

The trained model is `trainer.model`. Its `act` method takes a *batch* of observations, so add a batch dimension to play a single env:

```python
import gymnasium as gym

def play(model, seed=123):
    env = gym.make('CartPole-v1')
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        action = model.act(obs[None], deterministic=True).action   # batch of 1
        obs, reward, terminated, truncated, _ = env.step(int(action[0]))
        total += reward
        done = terminated or truncated
    env.close()
    return total

print(play(trainer.model))
```

`deterministic=True` takes the most likely action, as evaluation does, rather than sampling one.

To reload it later, for example in a new session, build a trainer from the same config and call `resume()`. This loads the latest checkpoint from `checkpoint_dir`: the weights, the optimizer state, and the training progress.

```python
with PPOTrainer(cfg) as trainer:
    trainer.resume()
    print(trainer.evaluate())        # the trained agent's return
```

To carry on training instead, raise the budget and set `resume=True`, so the first `fit()` resumes before training. Because the step count is restored too, training continues from step 100,000 instead of starting again:

```python
longer = cfg.model_copy(update={'resume': True,
                                'model': cfg.model.model_copy(update={'total_timesteps': 150_000})})
with PPOTrainer(longer) as trainer:
    history = trainer.fit()          # 50,000 more steps
```

## The whole script

```python
from dragon_torch.data.envs import DragonEnvConfig
from dragon_torch.models.RL import PPOConfig
from dragon_torch.trainers.RL import EvalConfig, PPOTrainer, PPOTrainerConfig

cfg = PPOTrainerConfig(
    env=DragonEnvConfig(env_id='CartPole-v1', n_envs=8, seed=0),
    model=PPOConfig(checkpoint_dir='runs/cartpole', save_every=10, total_timesteps=100_000),
    eval=EvalConfig(every_steps=10_000, n_episodes=10))

with PPOTrainer(cfg) as trainer:
    history = trainer.fit()
    print(trainer.evaluate())
```

## Next steps

- **Try another algorithm.** Swap in `A2CTrainer` and `A2CTrainerConfig` with `model=A2CConfig(...)`. See [RL trainers](rl_trainers.md).
- **Try another env.** Change `env_id`, or shape the reward with a wrapper. See [Environments](envs.md).
- **Tune the hyperparameters.** `space_for(PPOTrainer)` gives a ready-made search space. See [Tuning](tuning.md).
- **Train several agents together.** See [Multi-agent RL](multi_agent.md).
