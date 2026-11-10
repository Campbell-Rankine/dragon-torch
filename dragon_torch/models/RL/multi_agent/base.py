"""DragonRLModel for multi-agent envs. Agents are handled in groups (agents
that share one network): every obs / action / value crossing the Actor
protocol is a {group: batch} dict, batched [n_envs, agents_in_group, ...].
Which agents form which group is the env layer's business (AgentGroups); the
model only sees groups."""
from typing import Any, Callable, Mapping
import numpy as np
import torch
from dragon_torch.common.torch.tree import Tree, tree_map
from dragon_torch.data.RL.collector import ActOutput
from dragon_torch.modules.RL.heads import deterministic_action
from dragon_torch.modules.RL.policy import MultiAgentPolicyModule
from ..base import DragonRLConfig, DragonRLModel


class MultiAgentRLConfig(DragonRLConfig):
    kind: str = 'marl-model'


def _leaves(tree: Tree) -> list[Any]:
    out: list[Any] = []
    tree_map(out.append, tree)
    return out


def lead_shape(tree: Tree, ndim: int = 2) -> tuple[int, ...]:
    """The leading `ndim` dims shared by every leaf of an obs tree."""
    return tuple(_leaves(tree)[0].shape[:ndim])


def merge_lead(tree: Tree) -> Tree:
    """[N, n, ...] leaves -> [N * n, ...]."""
    return tree_map(lambda x: x.reshape(-1, *x.shape[2:]), tree)


class MultiAgentRLModel(DragonRLModel):
    """DragonRLModel over a MultiAgentPolicyModule.

    act({group: obs [N, n_g, ...]}) -> ActOutput with
        action  {group: [N, n_g, ...]}
        extras  {'log_prob': {group: [N, n_g]}, 'value': {group: [N, n_g]}}
    value(obs) -> {group: [N, n_g]}

    Training batches are {'obs': {group: [B_g, ...]}, 'targets': {group: {...}}},
    already flattened over (env, agent) by MultiAgentRolloutSource; pair with
    MultiAgentLoss. evaluate() scores the team return (see
    MultiAgentEnvConfig.team_return) and keeps each group's mean per-agent
    return in last_eval as 'return/<group>'.
    """
    kind = 'marl-model'

    def __init__(self, module: MultiAgentPolicyModule, loss_fn: Callable, cfg: DragonRLConfig):
        super().__init__(module, loss_fn, cfg)

    def _forward_grouped(self, obs: Mapping[str, Tree]) -> tuple[dict, dict[str, tuple[int, ...]]]:
        obs = self._to_device(dict(obs))
        lead = {g: lead_shape(o) for g, o in obs.items()}
        return self.forward({g: merge_lead(o) for g, o in obs.items()}), lead

    @torch.no_grad()
    def act(self, obs: Mapping[str, Tree], deterministic: bool = False) -> ActOutput:
        self.eval()
        out, lead = self._forward_grouped(obs)
        actions, log_probs, values = {}, {}, {}
        for g, po in out.items():
            action = deterministic_action(po.dist) if deterministic else po.dist.sample()
            actions[g] = action.reshape(*lead[g], *action.shape[1:]).cpu()
            log_probs[g] = po.dist.log_prob(action).reshape(lead[g]).cpu()
            if po.value is not None:
                values[g] = po.value.reshape(lead[g]).cpu()
        extras: dict[str, Any] = {'log_prob': log_probs}
        if len(values) == len(out):
            extras['value'] = values
        return ActOutput(actions, extras)

    @torch.no_grad()
    def value(self, obs: Mapping[str, Tree]) -> dict[str, torch.Tensor]:
        self.eval()
        out, lead = self._forward_grouped(obs)
        missing = [g for g, po in out.items() if po.value is None]
        if missing:
            raise ValueError(f"Policy module returned no value for groups {missing}; "
                             "they need a critic head")
        return {g: po.value.reshape(lead[g]).cpu() for g, po in out.items()}

    def episode_stats(self, episodes: list[dict[str, Any]]) -> dict[str, float]:
        stats = super().episode_stats(episodes)
        groups = {g for ep in episodes for g in ep.get('group_returns', {})}
        for g in sorted(groups):
            stats[f'return/{g}'] = float(np.mean([ep['group_returns'][g] for ep in episodes
                                                  if g in ep.get('group_returns', {})]))
        return stats
