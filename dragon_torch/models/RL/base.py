import math
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Literal, Optional, Sequence
import numpy as np
import torch
from pydantic import Field
from dragon_torch.data.RL.collector import ActOutput
from dragon_torch.models.base import Batch, DataLike, DragonModel, DragonModelConfig, EpochCallback
from dragon_torch.modules.RL.policy import DragonPolicyModule


class DragonRLConfig(DragonModelConfig):
    kind: str = 'rl-model'
    input_key: str = 'obs'
    target_key: Optional[str] = 'targets'
    # used only when neither `epochs` nor total_timesteps decides the length
    max_epochs: int = Field(default=1, ge=1)
    # env-step budget: fit() runs ceil(remaining / source.steps_per_epoch) epochs,
    # counting steps already taken (so a resumed run finishes the same budget)
    total_timesteps: Optional[int] = Field(default=None, ge=1)
    max_grad_norm: Optional[float] = Field(default=0.5, gt=0)
    # which return evaluate() scores: the env's own reward ('raw', the task) or
    # the reward after the env config's wrappers ('shaped', what the agent trains on)
    eval_return: Literal['raw', 'shaped'] = 'raw'


class DragonRLModel(DragonModel):
    """Abstract policy-gradient model (PPO / A2C, chosen by the loss) over a
    DragonPolicyModule. SingleAgentRLModel and MultiAgentRLModel implement the
    Actor protocol (`act`, `value`) for their kind of env; everything else is
    shared.

    Same trainer interface as DragonModel; RL-specific behaviour:
      act / value   the Actor protocol RL sources call to collect experience
      step          adds grad-norm clipping and records loss components
      evaluate      mean episode return over an EpisodeSource (higher is better)
      fit           epochs default from cfg.total_timesteps and the source's
                    steps_per_epoch
    """
    higher_is_better = True
    eval_metric = 'val_return'

    def __init__(self, module: DragonPolicyModule, loss_fn: Callable, cfg: DragonRLConfig):
        super().__init__(module, loss_fn, cfg)
        self._loss_metrics: dict[str, list[float]] = defaultdict(list)
        # means of the last evaluate(): {'return', 'shaped_return', 'length', ...}
        self.last_eval: dict[str, float] = {}

    # ------- data ------- #
    def _to_device(self, x: Any) -> Any:
        x = super()._to_device(x)
        # gym envs often emit float64; torch modules are float32
        if isinstance(x, torch.Tensor) and x.dtype == torch.float64:
            return x.float()
        return x

    # ------- Actor protocol ------- #
    @abstractmethod
    def act(self, obs: Any, deterministic: bool = False) -> ActOutput:
        """Sample (or pick greedily) actions for a batch of env obs; extras
        must carry 'log_prob' and, with a critic, 'value'."""

    @abstractmethod
    def value(self, obs: Any) -> Any:
        """State-value estimates for a batch of env obs."""

    # ------- training ------- #
    def step(self, batch: Batch, optimizer: torch.optim.Optimizer) -> float:
        self.train()
        optimizer.zero_grad()
        loss = self._batch_loss(batch)
        loss.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
        optimizer.step()
        for k, v in getattr(self.loss_fn, 'last_metrics', {}).items():
            self._loss_metrics[k].append(v)
        return loss.item()

    def epoch_metrics(self) -> dict[str, float]:
        out = {k: float(np.mean(v)) for k, v in self._loss_metrics.items()}
        self._loss_metrics.clear()
        return out

    @torch.no_grad()
    def evaluate(self, data: DataLike) -> float:
        """Mean episode return over data (an EpisodeSource): the raw return by
        default, the shaped one with cfg.eval_return='shaped'. All the means are
        kept in `last_eval`."""
        self.bind(data)
        self.eval()
        episodes = list(self.iter_batches(data))
        if not episodes:
            raise ValueError("evaluate() got no episodes")
        self.last_eval = self.episode_stats(episodes)
        return self.last_eval['shaped_return' if self.config.eval_return == 'shaped' else 'return']

    def episode_stats(self, episodes: list[dict[str, Any]]) -> dict[str, float]:
        """Means over evaluation episodes, stored as `last_eval`."""
        return {
            'return': float(np.mean([ep['return'] for ep in episodes])),
            'shaped_return': float(np.mean([ep.get('shaped_return', ep['return']) for ep in episodes])),
            'length': float(np.mean([ep['length'] for ep in episodes if 'length' in ep] or [np.nan])),
        }

    def fit(self, train_data: DataLike, optimizer: torch.optim.Optimizer,
            val_data: Optional[DataLike] = None, epochs: Optional[int] = None,
            callbacks: Sequence[EpochCallback] = ()) -> list[dict[str, float]]:
        # bind first so a checkpoint's source state (incl. global_step) is restored
        # before the remaining step budget is worked out
        self.bind(train_data, train=True)
        if epochs is None and self.config.total_timesteps is not None:
            if not hasattr(train_data, 'steps_per_epoch'):
                raise ValueError("total_timesteps needs a source with steps_per_epoch")
            remaining = self.config.total_timesteps - getattr(train_data, 'global_step', 0)
            epochs = math.ceil(remaining / train_data.steps_per_epoch)
            if epochs <= 0:
                return []
        return super().fit(train_data, optimizer, val_data=val_data, epochs=epochs,
                           callbacks=callbacks)

