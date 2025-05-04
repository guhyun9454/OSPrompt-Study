import os
import sys
import argparse
import torch
import numpy as np
import random
from random import shuffle
from collections import OrderedDict
import dataloaders
from dataloaders.utils import *
from torch.utils.data import DataLoader
import learners
import time
import datetime
from learners.metric_vil import evaluate_till_now

class Trainer:

    def __init__(self, args, seed, metric_keys, save_keys):

        # process inputs
        self.seed = seed
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers
        
        # model load directory
        self.model_top_dir = args.log_dir

        # select dataset
        self.grayscale_vis = False
        self.top_k = 1
        if args.dataset == 'CIFAR10':
            Dataset = dataloaders.iCIFAR10
            num_classes = 10
            self.dataset_size = [32,32,3]
        elif args.dataset == 'CIFAR100':
            Dataset = dataloaders.iCIFAR100
            num_classes = 100
            self.dataset_size = [32,32,3]
        elif args.dataset == 'ImageNet_R':
            Dataset = dataloaders.iIMAGENET_R
            num_classes = 200
            self.dataset_size = [224,224,3]
            self.top_k = 1
        elif args.IL_mode == 'vil':
            # VIL 시나리오 데이터셋 추가
            from dataloaders.vil import VILScenario as Dataset
            num_classes = args.num_classes
            self.dataset_size = [224,224,3] 
        else:
            raise ValueError('Dataset not implemented!')

        # upper bound flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes

        # iDigits와 VIL 모드일 때는 tasks를 따로 구성하지 않음 (이미 build_continual_dataloader에서 처리됨)
        if args.IL_mode == 'vil':
            self.train_dataset = Dataset(args)
            self.test_dataset = self.train_dataset  # 같은 객체 공유 (load_dataset에서 train/val 구분)
            self.tasks = []
            self.class_indices_by_task = []
            self.num_tasks = args.num_tasks
            self.task_names = [str(i+1) for i in range(self.num_tasks)]
        else:
            # 기존 코드: task 분할 및 데이터셋 로드
            # load tasks
            class_order = np.arange(num_classes).tolist() # [0,1,2,...,num_classes-1]
            class_order_logits = np.arange(num_classes).tolist()
            if self.seed > 0 and args.rand_split:
                print('=============================================')
                print('Shuffling....')
                # print('pre-shuffle:' + str(class_order))
                random.seed(self.seed)
                random.shuffle(class_order)
                # print('post-shuffle:' + str(class_order))
                print('=============================================')
            self.tasks = []
            self.class_indices_by_task = []
            p = 0
            while p < num_classes:
                inc = args.other_split_size if p > 0 else args.first_split_size
                self.tasks.append(class_order[p:p+inc])
                self.class_indices_by_task.append(class_order_logits[p:p+inc])
                p += inc
            self.num_tasks = len(self.tasks)
            self.task_names = [str(i+1) for i in range(self.num_tasks)]


            self.num_tasks = len(self.task_names)

            # datasets and dataloaders
            k = 1 # number of transforms per image
            if args.model_name.startswith('vit'):
                resize_imnet = True
            else:
                resize_imnet = False

            train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, resize_imnet=resize_imnet)
            test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug, resize_imnet=resize_imnet)

            self.train_dataset = Dataset(args.dataroot, train=True, lab = True, tasks=self.tasks,
                                download_flag=True, transform=train_transform, 
                                seed=self.seed, rand_split=args.rand_split, validation=args.validation)
            self.test_dataset  = Dataset(args.dataroot, train=False, tasks=self.tasks,
                                    download_flag=False, transform=test_transform, 
                                    seed=self.seed, rand_split=args.rand_split, validation=args.validation)

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare the self.learner (model)
        self.learner_config = {'num_classes': num_classes,
                        'lr': args.lr,
                        'debug_mode': args.debug_mode == 1,
                        'momentum': args.momentum,
                        'weight_decay': args.weight_decay,
                        'schedule': args.schedule,
                        'schedule_type': args.schedule_type,
                        'model_type': args.model_type,
                        'model_name': args.model_name,
                        'optimizer': args.optimizer,
                        'gpuid': args.gpuid,
                        'memory': args.memory,
                        'temp': args.temp,
                        'out_dim': num_classes,
                        'overwrite': args.overwrite == 1,
                        'DW': args.DW,
                        'batch_size': args.batch_size,
                        'upper_bound_flag': args.upper_bound_flag,
                        'tasks': self.class_indices_by_task if len(self.class_indices_by_task) > 0 else None,
                        'top_k': self.top_k,
                        'prompt_param':[self.num_tasks,args.prompt_param],
                        'query': args.query,
                        'args' : args,
                        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
        
        # VIL 모드에서 사용할 정확도 행렬 초기화
        if args.IL_mode == 'vil':
            self.acc_matrix = np.zeros((self.num_tasks, self.num_tasks))

    def task_eval(self, t_index, local=False, task='acc'):

        val_name = self.task_names[t_index]
        # print('validation split name:', val_name)
        
        # eval
        self.test_dataset.load_dataset(t_index, train=False)
        test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        if local:
            return self.learner.validation(test_loader, task_in = self.class_indices_by_task[t_index] if hasattr(self, 'class_indices_by_task') and len(self.class_indices_by_task) > 0 else None, task_metric=task)
        else:
            return self.learner.validation(test_loader, task_metric=task)

    def train(self, avg_metrics):
    
        # temporary results saving
        temp_table = {}
        for mkey in self.metric_keys: temp_table[mkey] = []
        temp_dir = self.log_dir + '/temp/'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)

        # VIL 모드 확인 - loaders가 orig_loaders로 이름 변경됨
        is_vil_mode = hasattr(self.train_dataset, 'orig_loaders') and hasattr(self.train_dataset, 'class_mask')

        # for each task
        for i in range(self.num_tasks):

            # save current task index
            self.current_t_index = i

            # print name
            train_name = self.task_names[i]
            print('======================', train_name, '=======================')

            # load dataset for task
            if is_vil_mode:
                # VIL 시나리오: 이미 class_mask가 설정되어 있음
                task = self.train_dataset.class_mask[i] if hasattr(self.train_dataset, 'class_mask') else []
                self.train_dataset.load_dataset(i, train=True)
                # 첫 태스크에서만 출력 차원 설정 (이미 모든 클래스를 알고 있음)
                if i == 0:
                    self.add_dim = self.learner.out_dim
                    self.learner.add_valid_output_dim(self.add_dim)
                else:
                    self.add_dim = 0  # 이후 태스크에서는 출력 차원 확장 불필요
            else:
                # 기존 CIL 방식
                task = self.class_indices_by_task[i]
                if self.oracle_flag:
                    self.train_dataset.load_dataset(i, train=False)
                    self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
                    self.add_dim += len(task)
                else:
                    self.train_dataset.load_dataset(i, train=True)
                    self.add_dim = len(task)
                
                # add valid class to classifier
                self.learner.add_valid_output_dim(self.add_dim)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # load dataset with memory
            #self.train_dataset.append_coreset(only=False)

            # load dataloader
            if is_vil_mode:
                # VIL 모드에서는 데이터로더가 이미 로드됨
                train_loader = self.train_dataset.curr_loader
            else:
                train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # learn
            if is_vil_mode:
                self.test_dataset.load_dataset(i, train=False)
                test_loader = self.test_dataset.curr_loader
            else:
                self.test_dataset.load_dataset(i, train=False)
                test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
            
            model_save_dir = self.model_top_dir + '/models/task-'+self.task_names[i]+'/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            avg_train_time = self.learner.learn_batch(train_loader, self.train_dataset, model_save_dir, test_loader)

            # save model
            self.learner.save_model(model_save_dir)
            
            # VIL 시나리오에서 평가 실행
            if is_vil_mode:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f'{f"Evaluating on Task {i+1}/{self.num_tasks}":=^60}')

                stats = evaluate_till_now(
                    model=self.learner.model,
                    data_loader=self.train_dataset.orig_loaders,   # 전체 task loader 리스트
                    device=device,
                    task_id=i,
                    class_mask=self.train_dataset.class_mask,
                    acc_matrix=self.acc_matrix,                   # 누적 정확도 행렬
                    args=self.learner.config["args"],
                )

                # 중간 성능을 로그/저장하려면 필요에 따라 avg_metrics에도 기록
                avg_metrics["A_last"]["global"][i, 0] = stats["A_last"]
                avg_metrics["A_avg"]["global"][i, 0]  = stats["A_avg"]
                avg_metrics["Forgetting"]["global"][i, 0] = stats["Forgetting"]

            if avg_train_time is not None: avg_metrics['time']['global'][i] = avg_train_time

        return avg_metrics
    
    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):

        # unpack dictionary
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']
        avg_acc_pt_local = acc_dict['pt-local']

        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * self.num_tasks
        acc_matrix = np.zeros((self.num_tasks, self.num_tasks))
        for i in range(self.num_tasks):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i+1):
                val_name = self.task_names[j]
                acc_matrix[i,j] = acc_table[val_name][train_name]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[j,i,self.seed] = acc_table[val_name][train_name]
                avg_acc_pt_local[j,i,self.seed] = acc_table_pt[val_name][train_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)

        # get forget score
        f_score = self.cal_fscore(acc_matrix)

        # Gather the final avg accuracy
        avg_acc_all[:,self.seed] = avg_acc_history

        # repack dictionary and return
        return {'global': avg_acc_all,'pt': avg_acc_pt,'pt-local': avg_acc_pt_local}, f_score

    def evaluate(self, avg_metrics):
        # VIL 모드 확인
        is_vil_mode = hasattr(self.train_dataset, 'orig_loaders') and hasattr(self.train_dataset, 'class_mask')

        # for convenience
        if is_vil_mode:
            # 행렬 초기화
            acc_matrix = np.zeros((self.num_tasks, self.num_tasks))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f'Running on: {device}')
            
            # 모든 태스크에 대한 평가
            for t in range(self.num_tasks):
                self.test_dataset.load_dataset(t, train=False)
                test_loader = self.test_dataset.curr_loader
                for task_id in range(min(t+1, self.num_tasks)):
                    acc_matrix[t, task_id] = self.task_eval(task_id)

            # 평가 메트릭 계산
            stats = evaluate_till_now(
                model=self.learner.model,
                data_loader=[self.test_dataset.orig_loaders[t]['val'] for t in range(self.num_tasks)],
                device=device,
                task_id=self.num_tasks - 1,  # 마지막 태스크
                class_mask=self.test_dataset.class_mask,
                acc_matrix=acc_matrix,
                args=self.learner.args
            )
            
            # 평균 정확도 저장
            avg_metrics['A_last']['global'][-1, 0] = stats['A_last']
            avg_metrics['A_avg']['global'][-1, 0] = stats['A_avg']
            avg_metrics['Forgetting']['global'][-1, 0] = stats['Forgetting']
            
            # F-score 계산 및 반환
            f_score = self.cal_fscore(acc_matrix[:, -1])
            
            return avg_metrics, f_score
        else:
            # 기존 평가 방식
            # validation
            for t in range(self.num_tasks):
                if t > self.current_t_index: break
                name = self.task_names[t]
                for i, task in enumerate(self.metric_keys):
                    if task in ['time','A_last','A_avg','Forgetting']: continue
                    
                    curr_acc, curr_pt = self.task_eval(t, False, task=task), []
                    # pt mode
                    if hasattr(self.learner, 'valid_out_dim'):
                        # format pt results (correct, total)
                        for valid_t in range(len(self.learner.valid_out_dim)):
                            if valid_t > t: break
                            pt_acc, pt_local_acc = self.task_eval(t, task_t = valid_t, task=task)
                            curr_pt.append([pt_acc, pt_local_acc])
                    
                    # record accuracy
                    avg_metrics[task]['global'][t, 0] = curr_acc
                    # record pt accuracy
                    if len(curr_pt) > 0:
                        for v_i, pt_acc in enumerate(curr_pt):
                            avg_metrics[task]['pt'][t, v_i, 0] = pt_acc[0]
                            avg_metrics[task]['pt-local'][t, v_i, 0] = pt_acc[1]
            
            # F-score 계산 (1 차원 배열로 전달)
            f_score = self.cal_fscore(avg_metrics['acc']['global'].reshape(-1))
            
            return avg_metrics, f_score

    def cal_fscore(self, y):
        index = y.shape [1]
        fgt = 0
        for t in range(1, index):
            for i in range(t):
                fgt += (y[t-1,i]-y[t,i])*(1/t)

        fgt = fgt/(index-1)
        return fgt