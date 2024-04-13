import random
import yaml
import sys
import os
import time
import subprocess
import signal
from itertools import product


processes = []
def handler(signum, frame):
    for p in processes:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except:
            pass
    sys.exit()


def gpu_info(gpu_index):
    ## through nvidia-smi
    # info = os.popen('nvidia-smi|grep %').read().split('\n')[gpu_index]
    # rate = int(info.split('|')[3].strip().split('%')[0])
    # memory = int(info.split('|')[2].split()[0].split('M')[0])

    # through gpustat
    info = os.popen('gpustat|grep %').read().split('\n')[gpu_index]
    rate = int(info.split('|')[1].strip().split(',')[1].split('%')[0].strip())
    memory = int(info.split('|')[2].strip().split('/')[0].strip())
    return rate, memory


def get_gpu():
    for gid in gpus:
        util, memory = gpu_info(gid)
        if util <= max_util and memory <= max_memory:
            return gid
    return -1

if len(sys.argv) == 3 and sys.argv[2] == '-C':
    is_continue = True
    print('Continue Mode.')
else:
    is_continue = False

yaml_file = sys.argv[1]
d = open(yaml_file, 'r')
y = yaml.load(d, Loader=yaml.FullLoader)

exp_name = y['exp_name']
command = y['command']
params = y['params']
gpus = y['gpus']
sampling = y.get('sampling', 'grid')
max_trials = y.get('max_trials', 600)
wait_second = y.get('wait_second', 50)
max_util = y.get('max_util', 95)
max_memory = y.get('max_memory', 9000)

param_names = [p['name'] for p in params]
param_values = [p['values'] for p in params]
product_param_values = list(product(*param_values))
n_params = len(param_names)
if sampling == 'random':
    random.shuffle(product_param_values)
    product_param_values = product_param_values[:max_trials]

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, handler)
    for param in product_param_values:
        param_dict = {param_names[i]: param[i] for i in range(n_params)}
        cmd = command.format(**param_dict)
        run_name = '_'.join([f'{p[0]}_{p[1]}' for p in param_dict.items()])
        f_log = os.path.join("search", "log", exp_name, f"{run_name}.log")
        if not os.path.exists(os.path.dirname(f_log)):
            os.makedirs(os.path.dirname(f_log), exist_ok=True)

        if is_continue and os.path.exists(f_log):
            with open(f_log, 'r') as f:
                # TODO: must change to fit new output format
                if 'test Recall' in f.read():
                    continue

        stdout_file = open(f_log, 'w')
        stderr_file = open(f_log, 'w')
        while True:
            gpu_id = get_gpu()
            if gpu_id != -1:
                cmd = cmd + f' --gpu_id {gpu_id}'
                print(run_name)
                p = subprocess.Popen(cmd.split(), shell=False, preexec_fn=os.setsid, stdout=stdout_file, stderr=stderr_file)
                processes.append(p)
                time.sleep(wait_second)
                break
            time.sleep(2)
