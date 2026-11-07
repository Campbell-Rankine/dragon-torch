from typing import Any, Iterator, Optional
import torch
from common.torch.tree import tree_map, tree_stack
from data.envs.config import DragonEnvConfig
from data.envs.spaces import obs_to_tensor
from data.RL.collector import EnvCollector, ValueActor


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor,
                terminated: torch.Tensor, dones: torch.Tensor, gamma: float,
                gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over [T, N] tensors.

    next_values[t] is V of the *true* successor of step t (the final obs when an
    episode ended), so truncated episodes bootstrap while terminated ones don't,
    and the advantage recursion is cut at every episode boundary.
    Returns (advantages, returns).
    """
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * next_values[t] * (1 - terminated[t].float()) - values[t]
        running = delta + gamma * gae_lambda * (1 - dones[t].float()) * running
        advantages[t] = running
    return advantages, advantages + values


class RolloutSource:
    """On-policy data source for DragonRLModel.fit.

    Each iteration (= one training epoch) collects n_steps x n_envs fresh
    transitions with the bound model, computes GAE, then yields
    update_epochs x n_minibatches shuffled minibatches shaped
        {'obs': ..., 'targets': {'actions', 'log_prob', 'values', 'advantages', 'returns'}}.
    PPO: update_epochs ~4-10, n_minibatches ~4. A2C: update_epochs=1, n_minibatches=1.
    """

    def __init__(self, env_cfg: DragonEnvConfig, n_steps: int = 128, gamma: float = 0.99,
                 gae_lambda: float = 0.95, update_epochs: int = 4, n_minibatches: int = 4,
                 normalize_obs: bool = False, normalize_reward: bool = False):
        self.collector = EnvCollector(env_cfg, normalize_obs, normalize_reward, gamma)
        if (n_steps * self.collector.n_envs) < n_minibatches:
            raise ValueError("n_steps * n_envs must be >= n_minibatches")
        self.n_steps, self.gamma, self.gae_lambda = n_steps, gamma, gae_lambda
        self.update_epochs, self.n_minibatches = update_epochs, n_minibatches
        self.actor: Optional[ValueActor] = None
        self._rng = torch.Generator().manual_seed(env_cfg.seed)

    def bind(self, model: ValueActor) -> None:
        if not isinstance(model, ValueActor):
            raise TypeError(f"RolloutSource needs a model with act() and value(), got {type(model)}")
        self.actor = model

    @property
    def steps_per_epoch(self) -> int:
        return self.n_steps * self.collector.n_envs

    @property
    def global_step(self) -> int:
        return self.collector.global_step

    # ------- collection ------- #
    def collect(self) -> dict[str, Any]:
        """One rollout as [T, N, ...] tensors, plus advantages / returns."""
        if self.actor is None:
            raise RuntimeError("RolloutSource is unbound; fit()/evaluate() bind the model")
        obs, actions, rewards, terminated, dones, log_probs, values = [], [], [], [], [], [], []
        truncation_values = []  # (t, mask, V(final obs)) for truncated envs
        for t in range(self.n_steps):
            tr = self.collector.step(self.actor)
            missing = {'log_prob', 'value'} - set(tr.extras)
            if missing:
                raise KeyError(f"act() must return extras {sorted(missing)} for on-policy training")
            obs.append(obs_to_tensor(tr.obs))
            actions.append(tr.action)
            rewards.append(torch.as_tensor(tr.reward))
            terminated.append(torch.as_tensor(tr.terminated))
            dones.append(torch.as_tensor(tr.done))
            log_probs.append(tr.extras['log_prob'])
            values.append(tr.extras['value'])
            if tr.truncated.any():
                mask = torch.as_tensor(tr.truncated)
                truncation_values.append((t, mask, self.actor.value(tr.next_obs)))

        values_t = torch.stack(values).float()
        next_values = torch.cat([values_t[1:], self.actor.value(self.collector.obs)[None].float()])
        for t, mask, v in truncation_values:
            next_values[t] = torch.where(mask, v.float(), next_values[t])
        rewards_t = torch.stack(rewards)
        advantages, returns = compute_gae(rewards_t, values_t, next_values, torch.stack(terminated),
                                          torch.stack(dones), self.gamma, self.gae_lambda)
        return {'obs': tree_stack(obs), 'actions': torch.stack(actions),
                'log_prob': torch.stack(log_probs), 'values': values_t,
                'advantages': advantages, 'returns': returns, 'rewards': rewards_t}

    def __iter__(self) -> Iterator[dict[str, Any]]:
        data = self.collect()
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        obs = tree_map(flat, data['obs'])
        targets = {k: flat(data[k]) for k in ('actions', 'log_prob', 'values', 'advantages', 'returns')}
        total = self.steps_per_epoch
        size = total // self.n_minibatches
        for _ in range(self.update_epochs):
            perm = torch.randperm(total, generator=self._rng)
            for i in range(self.n_minibatches):
                idx = perm[i * size:(i + 1) * size]
                yield {'obs': tree_map(lambda x: x[idx], obs),
                       'targets': {k: v[idx] for k, v in targets.items()}}

    # ------- hooks ------- #
    def metrics(self) -> dict[str, float]:
        return self.collector.metrics()

    def state_dict(self) -> dict[str, Any]:
        return {'collector': self.collector.state_dict(), 'rng': self._rng.get_state()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.collector.load_state_dict(state['collector'])
        self._rng.set_state(state['rng'])

    def close(self) -> None:
        self.collector.close()
