"""DragonRLModel for single-agent (gymnasium) envs."""
from typing import Any, Callable
import torch
from dragon_torch.data.RL.collector import ActOutput
from dragon_torch.modules.RL.heads import deterministic_action
from dragon_torch.modules.RL.policy import SingleAgentPolicyModule
from ..base import DragonRLConfig, DragonRLModel


class SingleAgentRLModel(DragonRLModel):
    """DragonRLModel over a SingleAgentPolicyModule (e.g. ActorCritic) and a
    gymnasium env: obs, actions, log-probs and values are batched over envs."""
    kind = 'rl-model'

    def __init__(self, module: SingleAgentPolicyModule, loss_fn: Callable, cfg: DragonRLConfig):
        super().__init__(module, loss_fn, cfg)

    @torch.no_grad()
    def act(self, obs: Any, deterministic: bool = False) -> ActOutput:
        self.eval()
        out = self.forward(self._to_device(obs))
        action = deterministic_action(out.dist) if deterministic else out.dist.sample()
        extras = {'log_prob': out.dist.log_prob(action)}
        if out.value is not None:
            extras['value'] = out.value
        return ActOutput(action.cpu(), {k: v.cpu() for k, v in extras.items()})

    @torch.no_grad()
    def value(self, obs: Any) -> torch.Tensor:
        self.eval()
        out = self.forward(self._to_device(obs))
        if out.value is None:
            raise ValueError("Policy module returned no value; it needs a critic head")
        return out.value.cpu()
