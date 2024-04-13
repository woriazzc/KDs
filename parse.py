import os
import argparse

from utils.parse_utils import DictAction, parse_cfg
from utils import load_yaml

LOG_DIR = 'logs/'
CONFIG_DIR = 'configs/'
DATA_DIR = 'data/'


parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     add_help=False)
parser.add_argument('--run_all', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--no_log', action='store_true')
parser.add_argument('--suffix', type=str, default='')

parser.add_argument('--dataset', type=str, default='gowalla')
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--num_ns', type=int, default=1)

parser.add_argument('--backbone', type=str, default='bpr')
parser.add_argument('--model', type=str, default='rrd')

parser.add_argument('--train_teacher', action='store_true')

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
        help='override some settings in the used config file')

args = parser.parse_args()

"""
Merge yaml and cmd
    priority: cmd > yaml > parser.default
"""
# Read from configs/<dataset>/<model>.yaml
if args.train_teacher:
    args.model = "scratch"
config_path = os.path.join(CONFIG_DIR, args.dataset.lower(), args.backbone.lower(), f"{args.model.lower()}.yaml")
args = parse_cfg(args, config_path, args.cfg)
args.__dict__.update({"LOG_DIR": LOG_DIR, "CONFIG_DIR": CONFIG_DIR, "DATA_DIR": DATA_DIR})

"""
create args for backbone
"""
backbone_config = load_yaml(os.path.join(CONFIG_DIR, args.dataset.lower(), args.backbone.lower(), f"base_config.yaml"))
teacher_args = argparse.Namespace()
teacher_args.__dict__.update(backbone_config["teacher"])
student_args = argparse.Namespace()
student_args.__dict__.update(backbone_config["student"])
