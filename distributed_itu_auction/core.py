import os
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist

from batched_itu_auction import ITUauction as BatchedITUAuction
from distributed_itu_auction.config import DistributedConfig


def shard_bounds(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    if total < 0:
        raise ValueError("total must be nonnegative")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank is outside the process group")
    width, remainder = divmod(total, world_size)
    start = rank * width + min(rank, remainder)
    return start, start + width + (rank < remainder)


def _backend(config: DistributedConfig) -> str:
    if config.backend is not None:
        return config.backend
    if config.device is not None:
        return "nccl" if torch.device(config.device).type == "cuda" else "gloo"
    return "nccl" if torch.cuda.is_available() else "gloo"


def _process_group(config: DistributedConfig):
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if not dist.is_initialized():
        backend = _backend(config)
        if backend == "nccl":
            torch.cuda.set_device(_cuda_device(config))
        dist.init_process_group(backend=backend)
    return dist.group.WORLD


def _cuda_device(config: DistributedConfig, rank: int = 0) -> torch.device:
    requested = torch.device(config.device) if config.device is not None else None
    if requested is not None and requested.type != "cuda":
        raise ValueError("NCCL requires a CUDA device")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    index = requested.index if requested is not None else local_rank
    return torch.device("cuda", index)


def _device(config: DistributedConfig, rank: int, process_group) -> torch.device:
    backend = dist.get_backend(process_group)
    requested = torch.device(config.device) if config.device is not None else None
    if backend != "nccl":
        return requested or torch.device("cpu")
    device = _cuda_device(config, rank)
    torch.cuda.set_device(device)
    return device


class DistributedITUAuction:
    def __init__(
        self,
        config: DistributedConfig,
        get_U_t_i_j: Callable,
        get_V_t_i_j: Callable,
        process_group=None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.process_group = (
            process_group if process_group is not None else _process_group(config)
        )
        self.rank = dist.get_rank(self.process_group)
        self.world_size = dist.get_world_size(self.process_group)
        if self.world_size > config.num_t:
            raise ValueError("num_t must be at least the process-group size")
        self.t_start, self.t_stop = shard_bounds(config.num_t, self.rank, self.world_size)
        self.num_t_local = self.t_stop - self.t_start
        self.device = device or _device(config, self.rank, self.process_group)
        self.local_auction = BatchedITUAuction(
            config.num_i,
            config.num_j,
            self.num_t_local,
            get_U_t_i_j,
            get_V_t_i_j,
            device=self.device,
        )

    @property
    def local_t_range(self) -> range:
        return range(self.t_start, self.t_stop)

    def forward_auction(self, *args, **kwargs):
        return self.local_auction.forward_auction(*args, **kwargs)

    def reverse_auction(self, *args, **kwargs):
        return self.local_auction.reverse_auction(*args, **kwargs)

    def forward_reverse_scaling(self, eps_init, eps_target, scaling_factor):
        return self.local_auction.forward_reverse_scaling(
            eps_init,
            eps_target,
            scaling_factor,
        )

    def _gather_tensor(self, tensor: torch.Tensor, dst: int):
        if tensor.shape[0] != self.num_t_local:
            raise ValueError("tensor does not match the local problem shard")
        max_t = (self.config.num_t + self.world_size - 1) // self.world_size
        if self.num_t_local < max_t:
            padding = tensor.new_zeros((max_t - self.num_t_local, *tensor.shape[1:]))
            tensor = torch.cat((tensor, padding), dim=0)
        gathered = None
        if self.rank == dst:
            gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        global_dst = dist.get_global_rank(self.process_group, dst)
        dist.gather(tensor, gathered, dst=global_dst, group=self.process_group)
        if self.rank != dst:
            return None
        shards = []
        for rank, shard in enumerate(gathered):
            start, stop = shard_bounds(self.config.num_t, rank, self.world_size)
            shards.append(shard[: stop - start])
        return torch.cat(shards, dim=0)

    def gather_results(self, u_t_i, v_t_j, mu_t_i_j, dst=0):
        return (
            self._gather_tensor(u_t_i, dst),
            self._gather_tensor(v_t_j, dst),
            self._gather_tensor(mu_t_i_j, dst),
        )

    def check_equilibrium(self, u_t_i, v_t_j, mu_t_i_j, eps=0):
        auction = self.local_auction
        utilities = auction.get_U_t_i_j(
            v_t_j,
            auction.all_t.unsqueeze(1),
            auction.all_i.unsqueeze(0),
        )
        cs = (utilities.amax(dim=2) - u_t_i).amax()
        feasible = torch.all(mu_t_i_j.sum(dim=2) <= 1) & torch.all(
            mu_t_i_j.sum(dim=1) <= 1
        )
        ir_i = torch.all(
            u_t_i[mu_t_i_j.sum(dim=2) == 0] <= auction.u_0 + eps
        )
        ir_j = torch.all(
            v_t_j[mu_t_i_j.sum(dim=1) == 0] <= auction.v_0 + eps
        )
        dist.all_reduce(cs, op=dist.ReduceOp.MAX, group=self.process_group)
        flags = torch.stack((feasible, ir_i, ir_j)).to(dtype=torch.uint8)
        dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=self.process_group)
        return (
            cs.item(),
            bool(flags[0].item()),
            bool(flags[1].item()),
            bool(flags[2].item()),
        )
