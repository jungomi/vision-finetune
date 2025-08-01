import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

from utils import nested_dict
from utils.nested_dict import NestedDict


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    if not is_dist():
        return 1
    return dist.get_world_size()


def rank() -> int:
    if not is_dist():
        return 0
    return dist.get_rank()


def is_main() -> bool:
    return rank() == 0


def init(
    rank: int,
    world_size: int,
    addr: str = "localhost",
    port: int = 12345,
    device_ids: list[int] | None = None,
):
    """
    Initialise the distributed group for the processes.

    Also sets the appropriate device to use, which is either the rank or which ever
    device id is assigned to that rank.

    Args:
        rank (int): Rank of the process
        world_size (int): Number of total processes in the group
        addr (str): Address to communicate between the processes [Default: localhost]
        port (int): Port for the communication [Default: 12345]
        device_ids (list[int], optional): Assign different device ids for each rank. If
            not specified, it will use the rank as the device id.

    Returns:
        device_id (int): Id of the device that is used.
    """
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method="env://",
    )
    device_id = rank if device_ids is None else device_ids[rank]
    torch.cuda.set_device(device_id)

    return device_id


@torch.no_grad()
def sync_tensor(tensor: torch.Tensor, reduction: str | None = "mean") -> torch.Tensor:
    """
    Synchronises the tensor across all processes.

    Args:
        tensor (torch.Tensor): Tensor to be synchronised.
        reduction (str): How to combine the results. When the reduction is
            "sum" or "mean" a single value will be created out of all the synchronised
            values. If the reduction is "none", the values of all processes will be
            given as an additional dimension at the beginning (dim=0).

    Returns:
        out (torch.Tensor): Synchronised tensor
    """
    num_procs = world_size()
    if num_procs == 1:
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(num_procs)]
    dist.all_gather(gathered, tensor)
    gathered_stack = torch.stack(gathered, dim=0)
    if reduction == "mean":
        return torch.mean(gathered_stack, dim=0)
    elif reduction == "sum":
        return torch.sum(gathered_stack, dim=0)
    elif reduction == "none" or reduction is None:
        return gathered_stack
    else:
        raise ValueError(
            f"reduction={repr(reduction)} is not supported, "
            'must be one of "mean" | "sum" | "none"'
        )


def sync_values(
    values: list[float], device: torch.device, reduction: str | None = "mean"
) -> list[float]:
    """
    Synchronises a list of simple values (numbers) across all processes.

    Args:
        values (list[float]): List of values to be synchronised.
        device (torch.device): Device on which the synchronised tensor should be placed.
        reduction (str): How to combine the results. When the reduction is
            "sum" or "mean" a single value will be created out of all the synchronised
            values. If the reduction is "none", the values of all processes will be
            given as a list.

    Returns:
        out (list[float]): Synchronised values
    """
    if world_size() == 1:
        return values
    values_tensor = torch.tensor(values, dtype=torch.float, device=device)
    return sync_tensor(values_tensor, reduction=reduction).tolist()


def sync_dict_values(
    d: NestedDict[float], device: torch.device, reduction: str | None = "mean"
) -> NestedDict[float]:
    """
    Synchronises a (nested) dictionary with simple values (numbers) across all
    processes.

    Args:
        d (NestedDict[float]): Dictionary to be synchronised. It can be nested as long as
            all leaf values can be stored in a tensor.
        device (torch.device): Device on which the synchronised tensor should be placed.
        reduction (str): How to combine the results. When the reduction is
            "sum" or "mean" a single value will be created out of all the synchronised
            values. If the reduction is "none", the values of all processes will be
            given as a list.

    Returns:
        out (NestedDict[float]): Synchronised dictionary
    """
    if world_size() == 1:
        return d
    # Sort the keys in case the insertion order was different across the processes.
    keys = sorted(nested_dict.nested_keys(d, keep_none=False))
    values: list = [nested_dict.get_recursive(d, k) for k in keys]
    values = sync_values(values, device=device, reduction=reduction)
    out: NestedDict[float] = {}
    for k, v in zip(keys, values):
        nested_dict.set_recursive(out, k, v)
    return out


@contextmanager
def on_main_first(
    enabled: bool = True, join: bool = True, device_ids: list[int] | None = None
):
    """
    A context manager for tasks that should be executed on the main process first and
    the others only execute it once the main process is done. This is especially useuful
    when the main process does all the loading/processing and writes the result to disk,
    such that all other processes can just read the processed data instead of having to
    process it themselves and doing wasteful work.

    Example:

    with on_main_first():
        processed_path = Path("processed.png")
        if processed_path.exists():
            img = Image.open(processed_path)
        else:
            img = expensive_processing("original.png")
            img.save(processed_path)

    In this example only the main process will do the `expensive_processing`, because
    the other processes wait to execute whatever is in the with statement until the main
    process is out of it.

    Args:
        enabled (bool): Whether the context manager is enabled. [Default: True]
        join (bool): Whether to join the processes, i.e. synchronise them so that all
            processes are "finished" with the work at the same time. This can be helpful
            if the other processes still need some work and they should be synchronised
            in time. If that is not a concern, it can be turned off.
            [Default: True]
        device_ids (List[int], optional): The list of devices this process is using. In
            the case of DDP it would be `device_ids=[rank]` just like in the DDP
            constructor. It is neeeded here because otherwise the barrier will copy the
            CUDA data to the GPU that matches the rank, so if rank != device_id, then it
            will just occupy extra memory for no reason. Rather strange, but that's just
            how it is, so better avoid this.
    """
    # Every process except the main process waits here until the main process has
    # finished the task.
    if enabled and is_dist() and not is_main():
        dist.barrier(device_ids=device_ids)
    yield
    # The main process has finished and the others will be released from the barrier,
    # since all processes now triggered it.
    if enabled and is_dist() and is_main():
        dist.barrier(device_ids=device_ids)
    # Join them to ensure they finish at the same time.
    if enabled and join and is_dist():
        dist.barrier(device_ids=device_ids)
