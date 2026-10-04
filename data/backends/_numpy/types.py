import numpy as np
import torch
from typing import Union

type NumpyOnlyOutputDtypes = Union[np.ndarray, np.floating, np.integer]
type NumpyTorchOutputDtypes = Union[torch.FloatTensor,
                                    torch.IntTensor, torch.FloatTensor]
