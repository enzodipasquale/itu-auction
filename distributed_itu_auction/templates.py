from typing import Callable, Dict

import torch
import torch.distributed as dist

from distributed_itu_auction.config import DistributedConfig
from distributed_itu_auction.core import (
    DistributedITUAuction,
    _device,
    _process_group,
    shard_bounds,
)


def TU_template(phi_t_i_j: torch.Tensor, config: DistributedConfig):
    process_group = _process_group(config)
    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    if world_size > config.num_t:
        raise ValueError("num_t must be at least the process-group size")
    start, stop = shard_bounds(config.num_t, rank, world_size)
    local_size = stop - start
    if phi_t_i_j.ndim != 3:
        raise ValueError("phi_t_i_j must have shape (num_t, num_i, num_j)")
    if tuple(phi_t_i_j.shape[1:]) != (config.num_i, config.num_j):
        raise ValueError("phi_t_i_j does not match the configured market size")
    if phi_t_i_j.shape[0] == config.num_t:
        phi_t_i_j = phi_t_i_j[start:stop]
    elif phi_t_i_j.shape[0] != local_size:
        raise ValueError("phi_t_i_j must contain either the global or local shard")
    device = _device(config, rank, process_group)
    phi_t_i_j = phi_t_i_j.to(device)

    def get_u(v, t, i, j=None):
        if j is None:
            return phi_t_i_j[t, i] - v[t]
        return phi_t_i_j[t, i, j] - v

    def get_v(u, t, j, i=None):
        if i is None:
            if t.ndim > 1:
                t = t.reshape(-1)
                j = j.reshape(-1)
                return phi_t_i_j[t][:, :, j] - u[t].unsqueeze(2)
            return phi_t_i_j[t, :, j] - u[t]
        return phi_t_i_j[t, i, j] - u

    return DistributedITUAuction(
        config,
        get_u,
        get_v,
        process_group=process_group,
        device=device,
    )


_TEMPLATES: Dict[str, Callable] = {"TU": TU_template}


def get_distributed_template(name: str):
    try:
        return _TEMPLATES[name]
    except KeyError as error:
        raise ValueError("unknown distributed template: {}".format(name)) from error
