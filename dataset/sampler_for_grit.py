import math
from typing import TypeVar, Optional, Iterator, List 

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
import time

T_co = TypeVar('T_co', covariant=True)

# Just load the pre-defined index array
class Sampler_for_GRIT (Sampler[T_co]):
    def __init__(self, pre_indices: List[int] =None, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.pre_indices= pre_indices
        self.index_set_len= int(len(self.pre_indices)/self.num_replicas)

    def __iter__(self) -> Iterator[T_co]:
        # subsample
        indices = self.pre_indices[ self.rank*self.index_set_len:(self.rank+1)*self.index_set_len]

        return iter(indices)

    def __len__(self) -> int:
        return self.index_set_len

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
