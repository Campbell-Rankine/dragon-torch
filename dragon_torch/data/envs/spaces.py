"""Conversions between gymnasium spaces / batched env values and torch."""
from typing import Any, Union
import numpy as np
import torch
from gymnasium import spaces
from dragon_torch.common.torch.tree import tree_map

flatdim = spaces.flatdim


def obs_to_tensor(obs: Any, device: Union[str, torch.device] = 'cpu') -> Any:
    """Batched env obs (array / dict / tuple) -> tensors. Floats become float32 and
    bools float; integer obs (e.g. uint8 images, Discrete ids) keep their dtype so
    the module decides how to scale or embed them."""
    def leaf(x: Any) -> torch.Tensor:
        t = torch.as_tensor(np.asarray(x))
        if t.is_floating_point() or t.dtype == torch.bool:
            t = t.float()
        return t.to(device)
    return tree_map(leaf, obs)


def action_to_env(action: torch.Tensor, space: spaces.Space) -> np.ndarray:
    """Batched policy action -> what a vector env with single_action_space=space
    accepts. Box actions are clipped to bounds (the unclipped sample is what
    buffers store, so log-probs stay consistent); Discrete / MultiDiscrete
    indices are shifted by the space's `start`."""
    a = action.detach().cpu().numpy()
    if isinstance(space, spaces.Box):
        return np.clip(a, space.low, space.high).astype(space.dtype)
    if isinstance(space, spaces.Discrete):
        return (a.astype(np.int64) + int(space.start)).astype(space.dtype)
    if isinstance(space, spaces.MultiDiscrete):
        return (a.astype(np.int64) + space.start).astype(space.dtype)
    if isinstance(space, spaces.MultiBinary):
        return a.astype(space.dtype)
    raise NotImplementedError(
        f"No action conversion for {type(space).__name__}; wrap the env so it "
        "takes a Box / Discrete / MultiDiscrete / MultiBinary action")
