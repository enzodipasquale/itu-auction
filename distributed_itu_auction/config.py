from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass(frozen=True)
class DistributedConfig:
    num_i: int
    num_j: int
    num_t: int
    backend: Optional[str] = None
    device: Optional[Union[str, torch.device]] = None

    def __post_init__(self):
        if self.num_i < 2:
            raise ValueError("num_i must be at least 2")
        if self.num_j < 2:
            raise ValueError("num_j must be at least 2")
        if self.num_t < 1:
            raise ValueError("num_t must be positive")
        if self.backend not in (None, "gloo", "nccl"):
            raise ValueError("backend must be 'gloo' or 'nccl'")
        if self.device is not None and torch.device(self.device).type not in (
            "cpu",
            "cuda",
        ):
            raise ValueError("device must be CPU or CUDA")
