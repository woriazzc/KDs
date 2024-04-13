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
        self.log_path = os.path.join(args.LOG_DIR, args.dataset, args.backbone, args.model + ('_' if args.suffix != '' else '') + args.suffix + '.log')
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.no_log = no_log

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

        sys.stdout.write(string)
        sys.stdout.flush()
    
    def log_args(self, args, text="ARGUMENTS"):
        self.log('-' * 40 + text + '-' * 40, pre=False)
        for arg in vars(args):
            self.log('{:40} {}'.format(arg, getattr(args, arg)), pre=False)
        self.log('-' * 40 + text + '-' * 40, pre=False)
