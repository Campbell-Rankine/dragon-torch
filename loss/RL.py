"""Policy-gradient losses. Y_hat is a PolicyOutput; Y is the rollout `targets`
dict produced by RolloutSource. Each loss leaves its components in
`last_metrics`, which DragonRLModel averages into the epoch history."""
from typing import Any
import torch
from loss.base import DragonLossFunction


class RLLoss(DragonLossFunction):
    def __init__(self):
        super().__init__(None)
        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _require_value(Y_hat: Any) -> torch.Tensor:
        if Y_hat.value is None:
            raise ValueError(f"{type(Y_hat).__name__}.value is None; this loss needs a critic")
        return Y_hat.value

    @staticmethod
    def _normalize(adv: torch.Tensor) -> torch.Tensor:
        return (adv - adv.mean()) / (adv.std() + 1e-8) if adv.numel() > 1 else adv


class PPOLoss(RLLoss):
    """Clipped-surrogate PPO (Schulman et al. 2017) with optional value clipping."""
    kind = 'ppo'

    def __init__(self, clip_coef: float = 0.2, vf_coef: float = 0.5, ent_coef: float = 0.0,
                 clip_vloss: bool = True, normalize_advantages: bool = True):
        super().__init__()
        self.clip_coef, self.vf_coef, self.ent_coef = clip_coef, vf_coef, ent_coef
        self.clip_vloss, self.normalize_advantages = clip_vloss, normalize_advantages

    def compute(self, Y_hat: Any, Y: dict[str, torch.Tensor], **kw) -> torch.Tensor:
        value = self._require_value(Y_hat)
        log_prob = Y_hat.dist.log_prob(Y['actions'])
        log_ratio = log_prob - Y['log_prob']
        ratio = log_ratio.exp()
        adv = self._normalize(Y['advantages']) if self.normalize_advantages else Y['advantages']

        pg_loss = torch.max(-adv * ratio,
                            -adv * ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef)).mean()
        if self.clip_vloss:
            clipped = Y['values'] + (value - Y['values']).clamp(-self.clip_coef, self.clip_coef)
            v_loss = 0.5 * torch.max((value - Y['returns']) ** 2,
                                     (clipped - Y['returns']) ** 2).mean()
        else:
            v_loss = 0.5 * ((value - Y['returns']) ** 2).mean()
        entropy = Y_hat.dist.entropy().mean()

        with torch.no_grad():
            self.last_metrics = {
                'policy_loss': pg_loss.item(), 'value_loss': v_loss.item(),
                'entropy': entropy.item(),
                'approx_kl': ((ratio - 1) - log_ratio).mean().item(),
                'clip_frac': ((ratio - 1).abs() > self.clip_coef).float().mean().item(),
            }
        return pg_loss - self.ent_coef * entropy + self.vf_coef * v_loss


class A2CLoss(RLLoss):
    """Advantage actor-critic. Pair with RolloutSource(update_epochs=1, n_minibatches=1)."""
    kind = 'a2c'

    def __init__(self, vf_coef: float = 0.5, ent_coef: float = 0.0,
                 normalize_advantages: bool = False):
        super().__init__()
        self.vf_coef, self.ent_coef = vf_coef, ent_coef
        self.normalize_advantages = normalize_advantages

    def compute(self, Y_hat: Any, Y: dict[str, torch.Tensor], **kw) -> torch.Tensor:
        value = self._require_value(Y_hat)
        adv = self._normalize(Y['advantages']) if self.normalize_advantages else Y['advantages']
        pg_loss = -(adv.detach() * Y_hat.dist.log_prob(Y['actions'])).mean()
        v_loss = 0.5 * ((value - Y['returns']) ** 2).mean()
        entropy = Y_hat.dist.entropy().mean()
        self.last_metrics = {'policy_loss': pg_loss.item(), 'value_loss': v_loss.item(),
                             'entropy': entropy.item()}
        return pg_loss - self.ent_coef * entropy + self.vf_coef * v_loss
