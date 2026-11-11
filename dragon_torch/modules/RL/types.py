import torch
import numpy as np
from typing import Union

type SupportedRLBackends = Union[torch.Tensor, np.ndarray]
# runtime counterpart of SupportedRLBackends, for isinstance / issubclass checks
SUPPORTED_RL_BACKENDS: tuple[type, ...] = (torch.Tensor, np.ndarray)
