
import torch
from torch import distributed as dist

def get_dist_info():
    TORCH_VERSION = torch.__version__
    if TORCH_VERSION < "1.0":
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size