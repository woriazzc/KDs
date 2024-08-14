import os
import argparse

from utils.parse_utils import DictAction, parse_cfg
from utils import load_yaml

LOG_DIR = 'logs/'
CONFIG_DIR = 'configs/'
DATA_DIR = 'data/'
CKPT_DIR = 'checkpoints/'


parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     add_help=False)
parser.add_argument('--run_all', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--no_log', action='store_true')
parser.add_argument('--no_save', action='store_true')
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--verbose', action='store_true', help="For mlflow")
parser.add_argument('--ablation', action='store_true', help="perform ablation")

parser.add_argument('--task', type=str, choices=['rec, ctr'], default='rec')
parser.add_argument('--dataset', type=str, default='gowalla')
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--num_ns', type=int, default=1)
parser.add_argument('--neg_sampling_on_all', action='store_true')

parser.add_argument('--backbone', type=str, default='bpr')
parser.add_argument('--model', type=str, default='rrd')

parser.add_argument('--train_teacher', action='store_true')
parser.add_argument('--ckpt_interval', type=int, default=-1, help="number of interval epochs to store teacher's checkpoints, -1 for only save the best_epoch")

parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=0.)

parser.add_argument('--lmbda', type=float, default=1., help="weight of kd loss")

parser.add_argument('--eval_period', type=int, default=1)
parser.add_argument('--K_list', type=list, default=[10, 20])
parser.add_argument('--early_stop_metric', type=str, default='NDCG')
parser.add_argument('--early_stop_K', type=int, default=10)
parser.add_argument('--early_stop_patience', type=int, default=30)

parser.add_argument(
        '--cfg',
        nargs='+',
        action=DictAction,
        help='override some settings in the model config')
parser.add_argument(
        '--teacher',
        nargs='+',
        action=DictAction,
        help='override some settings in the teacher config')
parser.add_argument(
        '--student',
        nargs='+',
        action=DictAction,
        help='override some settings in the student config')

args = parser.parse_args()
if args.train_teacher:
    args.model = "scratch"
args.__dict__.update({"LOG_DIR": LOG_DIR, "CONFIG_DIR": CONFIG_DIR, "DATA_DIR": DATA_DIR, "CKPT_DIR": CKPT_DIR})

"""
create args for backbone
"""
teacher_args = argparse.Namespace()
student_args = argparse.Namespace()
"""
Merge yaml and cmd
    priority: cmd > yaml > parser.default
"""
model_config = load_yaml(os.path.join(CONFIG_DIR, args.dataset.lower(), args.backbone.lower(), f"{args.model.lower()}.yaml"))
if args.model == "scratch":
    key = "teacher" if args.train_teacher else "student"
    model_config = model_config[key]
backbone_config = load_yaml(os.path.join(CONFIG_DIR, args.dataset.lower(), args.backbone.lower(), f"base_config.yaml"))
args = parse_cfg(args, model_config, args.cfg)
teacher_args = parse_cfg(teacher_args, backbone_config["teacher"], args.teacher)
student_args = parse_cfg(student_args, backbone_config["student"], args.student)
args.__dict__.pop("cfg", None)
args.__dict__.pop("teacher", None)
args.__dict__.pop("student", None)
