from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import argparse
import torch
import numpy as np
import yaml
import json
import random
from trainer import Trainer
from continual_datasets.dataset_utils import set_data_config

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_args():
    
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()

    # Standard Args
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                         help="The list o f gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--log_dir', type=str, default="outputs/out",
                         help="Save experiments results in dir for future plotting!")
    parser.add_argument('--learner_type', type=str, default='default', help="The type (filename) of learner")
    parser.add_argument('--learner_name', type=str, default='NormalNN', help="The class name of learner")
    parser.add_argument('--query', type=str, default='vit', help="choose one of [poolformer]")
    parser.add_argument('--debug_mode', type=int, default=0, metavar='N',
                        help="activate learner specific settings for debug_mode")
    parser.add_argument('--overwrite', type=int, default=0, metavar='N', help='Train regardless of whether saved model exists')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--wandb_run', type=str, default=None, help='Wandb run name')
    parser.add_argument('--wandb_project', type=str, default=None, help='Wandb project name')

    # Model Args
    parser.add_argument('--model_type', type=str, default='zoo', help="Base model architecture type")
    parser.add_argument('--model_name', type=str, default='vit_pt_imnet', help="Base model architecture name")
    # parser.add_argument('--in_channels', type=int, default=3, help="Input channel dimension")
    # parser.add_argument('--out_dim', type=int, default=None, help="Output dimension")
    parser.add_argument('--optimizer', type=str, default='Adam', help="Optimizer to use (sgd, adam, etc.)")
    
    # Dataset Args    
    parser.add_argument('--dataset', type=str, default='CIFAR100', help="Dataset to use")
    parser.add_argument('--dataroot', type=str, default='/local_datasets', help="Path to data directory")
    parser.add_argument('--train_aug', default=True, action='store_true', help="Whether to use data augmentation")
    parser.add_argument('--validation', default=False, action='store_true', help="Split validation data from training")
    parser.add_argument('--rand_split', default=False, action='store_true', help="Randomize data splits")
    parser.add_argument('--first_split_size', type=int, default=20, help="Size of first task")
    parser.add_argument('--other_split_size', type=int, default=20, help="Size of other tasks")
    
    # Learning Args
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--workers', type=int, default=4, help="Number of workers for data loading")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs per task")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--schedule', nargs="+", type=int, default=[5], help="Learning rate schedule")
    parser.add_argument('--schedule_type', type=str, default='decay', help="Type of learning rate schedule")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum")
    parser.add_argument('--weight_decay', type=float, default=0.0005, help="Weight decay")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loader")

    # VIL 시나리오를 위한 인자 추가
    parser.add_argument('--IL_mode', type=str, default='cil', choices=['cil', 'dil', 'vil', 'joint'])
    parser.add_argument('--num_tasks', type=int, default=5, help="Number of tasks for incremental learning")
    parser.add_argument('--shuffle', type=int, default=False, help="Whether to shuffle class order")
    parser.add_argument('--develop_tasks', action='store_true', help="Print additional task information for debugging")
    parser.add_argument('--verbose', action='store_true', help="Print verbose information")
    parser.add_argument('--data_path', type=str, default='/local_datasets', help="Path to data directory")

    # CL Args          
    parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    parser.add_argument('--upper_bound_flag', default=False, action='store_true', help='Upper bound')
    parser.add_argument('--memory', type=int, default=0, help="size of memory for replay")
    parser.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')
    parser.add_argument('--prompt_param', nargs="+", type=float, default=[1, 1, 1],
                         help="e prompt pool size, e prompt length, g prompt length")

    # Config Arg (optional)
    parser.add_argument('--config', type=str, default="", help="Optional yaml config file")
    
    #Misc Args
    parser.add_argument('--develop', action='store_true', default=False)
    return parser

def get_args(argv):
    parser = create_args()
    args = parser.parse_args(argv)
    
    # 선택적으로 config 파일 로드 (제공된 경우)
    if args.config and os.path.exists(args.config):
        config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
        # 명령줄 인자가 우선권을 가지도록 config 먼저 업데이트한 후 args로 덮어씁니다
        temp_args = vars(args).copy()
        temp_args.update(config)
        args = argparse.Namespace(**temp_args)
    
    # VIL 모드를 위한 추가 설정
    if args.IL_mode == 'vil':
        # dataroot를 data_path로 설정
        args.dataroot = args.data_path
        set_data_config(args)
    
    return args

# want to save everything printed to outfile
class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    print(args)

    # Set seed for reproducibility
    seed_everything(args.seed)

    # wandb 초기화
    if args.wandb_run and args.wandb_project:
        import wandb
        import getpass
        
        args.wandb = True
        wandb.init(entity="OODVIL", project=args.wandb_project, name=args.wandb_run, config=args)
        wandb.config.update({"username": getpass.getuser()})
    else:
        args.wandb = False

    # duplicate output stream to output file
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    log_out = args.log_dir + '/output.log'
    sys.stdout = Logger(log_out)

    # save args
    with open(args.log_dir + '/args.yaml', 'w') as yaml_file:
        yaml.dump(vars(args), yaml_file, default_flow_style=False)
    
    # VIL 시나리오에서는 A_last와 A_avg 메트릭도 추가
    metric_keys = ['acc','time',]
    if args.IL_mode == 'vil':
        metric_keys.extend(['A_last', 'A_avg', 'Forgetting'])
    
    save_keys = ['global', 'pt', 'pt-local']
    global_only = ['time']
    avg_metrics = {}
    for mkey in metric_keys: 
        avg_metrics[mkey] = {}
        for skey in save_keys: avg_metrics[mkey][skey] = []

    # set up a trainer
    trainer = Trainer(args, args.seed, metric_keys, save_keys)

    # init total run metrics storage
    num_tasks = trainer.num_tasks
    for mkey in metric_keys: 
        avg_metrics[mkey]['global'] = np.zeros((num_tasks, 1))
        if (not (mkey in global_only)):
            avg_metrics[mkey]['pt'] = np.zeros((num_tasks, num_tasks, 1))
            avg_metrics[mkey]['pt-local'] = np.zeros((num_tasks, num_tasks, 1))

    # train model
    avg_metrics = trainer.train(avg_metrics)  

    # evaluate model
    avg_metrics, f_score = trainer.evaluate(avg_metrics)

    # save results
    for mkey in metric_keys:
        m_dir = args.log_dir+'/results-'+mkey+'/'
        if not os.path.exists(m_dir): os.makedirs(m_dir)
        for skey in save_keys:
            if (not (mkey in global_only)) or (skey == 'global'):
                save_file = m_dir+skey+'.yaml'
                result = avg_metrics[mkey][skey]
                yaml_results = {}
                if len(result.shape) > 2:
                    yaml_results['mean'] = result[:,:,0].tolist()
                    yaml_results['history'] = result[:,:,0].tolist()
                else:
                    yaml_results['mean'] = result[:,0].tolist()
                    yaml_results['history'] = result[:,0].tolist()
                with open(save_file, 'w') as yaml_file:
                    yaml.dump(yaml_results, yaml_file, default_flow_style=False)

    # Print the summary
    print('===Summary of experiment===')
    for mkey in metric_keys: 
        print(mkey, ' | mean:', avg_metrics[mkey]['global'][-1,0])
    
    # 시간 정보 출력 (h:m:s 형식)
    if 'total_time' in avg_metrics:
        total_seconds = avg_metrics['total_time']['global'][0, 0]
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        print(f'전체 학습 시간: {hours:02d}:{minutes:02d}:{seconds:02d}')
    
    print('F-score:', f_score)
    
    # wandb에 최종 결과 로깅
    if args.wandb:
        import wandb
        # 모든 최종 메트릭 로깅
        final_metrics = {}
        for mkey in metric_keys:
            if mkey != 'time':  # 시간은 제외
                metric_value = avg_metrics[mkey]['global'][-1,0]
                final_metrics[f'Final_{mkey}'] = metric_value
        
        # F-score 추가
        final_metrics['Final_F_score'] = f_score
        
        # 총 학습 시간 추가
        if 'total_time' in avg_metrics:
            final_metrics['Total_Time'] = avg_metrics['total_time']['global'][0, 0]
        
        wandb.log(final_metrics)
        
        # wandb 종료
        wandb.finish()
    


