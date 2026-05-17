import os
import sys
from datetime import date, datetime
import random
import yaml
import pyro
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class TransparentDistributedDataParallel(DDP):
    """DDP wrapper that exposes wrapped module attributes."""
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class ShardedChunkLoader:
    """Shard chunk-path iterable for DDP CTR training."""
    def __init__(self, chunk_paths, rank, world_size):
        usable = (len(chunk_paths) // world_size) * world_size
        if usable == 0:
            raise ValueError(
                f"Not enough CTR chunks ({len(chunk_paths)}) for world_size={world_size}. "
                "Please decrease nproc_per_node or increase batch/chunks."
            )
        self.chunk_paths = chunk_paths[:usable][rank:usable:world_size]

    def __len__(self):
        return len(self.chunk_paths)

    def __iter__(self):
        for chunk_path in self.chunk_paths:
            yield torch.load(chunk_path, map_location="cpu", weights_only=True)


def init_distributed(args):
    has_torchrun_env = all(k in os.environ for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK"])
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.ddp = has_torchrun_env and env_world_size > 1
    args.rank = 0
    args.local_rank = 0
    args.world_size = 1
    if not getattr(args, "ddp", False):
        return
    args.rank = int(os.environ["RANK"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    if args.rank != 0:
        args.no_log = True
        args.no_save = True


def cleanup_distributed(args):
    if getattr(args, "ddp", False) and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_dist():
    return dist.is_available() and dist.is_initialized()


def build_train_loader(dataset, batch_size, shuffle, args):
    if getattr(args, "ddp", False):
        sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=shuffle
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
        return loader, sampler
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, None


def set_sampler_epoch(sampler, epoch):
    if sampler is not None:
        sampler.set_epoch(epoch)


def shard_ctr_train_loader(loader, args):
    if not getattr(args, "ddp", False):
        return loader
    if not hasattr(loader, "chunk_paths"):
        return loader
    return ShardedChunkLoader(loader.chunk_paths, args.rank, args.world_size)


def maybe_parallelize(model, args):
    if not getattr(args, "ddp", False):
        return model
    return TransparentDistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False,
        broadcast_buffers=False
    )


def to_np(x):
    return x.detach().data.cpu().numpy()


def load_yaml(path):
    return yaml.load(open(path, "r"), Loader=yaml.FullLoader)


def seed_all(seed:int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def avg_dict(eval_dicts, final_dict=None):
    if final_dict is None:
        final_dict = {}
    flg_dict = eval_dicts[0]
    for k in flg_dict:
        if isinstance(flg_dict[k], dict):
            final_dict[k] = avg_dict([eval_dict[k] for eval_dict in eval_dicts])
        else:
            final_dict[k] = 0
            for eval_dict in eval_dicts:
                final_dict[k] += eval_dict[k]
            final_dict[k] /= len(eval_dicts)
    return final_dict


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


class Logger:
    def __init__(self, args, no_log):
        if args.model.lower() == "scratch":
            self.log_path = os.path.join(args.LOG_DIR, args.dataset, args.S_backbone, args.model + ('_' if args.suffix != '' else '') + args.suffix + '.log')
        else:
            self.log_path = os.path.join(args.LOG_DIR, args.dataset, args.S_backbone, args.T_backbone, args.model + ('_' if args.suffix != '' else '') + args.suffix + '.log')
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.no_log = no_log
        self.is_main_process = getattr(args, "rank", 0) == 0

    def log(self, content='', pre=True, end='\n'):
        string = str(content)
        if len(string) == 0:
            pre = False
        if pre:
            today = date.today()
            today_date = today.strftime("%m/%d/%Y")
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            string = today_date + "," + current_time + ": " + string
        string = string + end

        if not self.no_log:
            with open(self.log_path, 'a') as logf:
                logf.write(string)

        if self.is_main_process:
            sys.stdout.write(string)
            sys.stdout.flush()
    
    def log_args(self, args, text="ARGUMENTS"):
        self.log('-' * 40 + text + '-' * 40, pre=False)
        for arg in vars(args):
            self.log('{:40} {}'.format(arg, getattr(args, arg)), pre=False)
        self.log('-' * 40 + text + '-' * 40, pre=False)
