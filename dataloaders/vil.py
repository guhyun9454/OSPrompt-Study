from continual_datasets.build_incremental_scenario import build_continual_dataloader
import torch

class _TaskDataset(torch.utils.data.Dataset):
    """(img, label) → (img, label, task_id) 로 확장"""
    def __init__(self, base_ds, task_id):
        self.base_ds = base_ds
        self.task_id = task_id
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        return x, y, self.task_id

class VILScenario:
    """
    Trainer가 expect 하는 형태로 동작하기 위한 래퍼
    self.load_dataset(t, train=True/False) 호출 시
    내부 data_loader[t]['train' | 'val'] 반환
    """
    def __init__(self, args):
        self.orig_loaders, self.class_mask, self.domain_list = build_continual_dataloader(args)
        self.args = args
        self.num_tasks = len(self.orig_loaders)
        self.num_classes = args.num_classes
        self.curr_loader = None
        
    def _wrap(self, loader, task_id, train=True):
        ds = _TaskDataset(loader.dataset, task_id)
        sampler = (torch.utils.data.RandomSampler(ds) if train
                   else torch.utils.data.SequentialSampler(ds))
        return torch.utils.data.DataLoader(
            ds,
            batch_size   = loader.batch_size,
            sampler      = sampler,
            num_workers  = loader.num_workers,
            pin_memory   = True,
        )
        
    def load_dataset(self, t, train=True):
        """
        기존 iDataset과 동일한 시그너처로 동작하여 호환성 유지
        """
        key = 'train' if train else 'val'
        base_loader = self.orig_loaders[t][key]
        self.curr_loader = self._wrap(base_loader, t, train)
        return self.curr_loader
        
    def __len__(self):
        if self.curr_loader is None:
            return 0
        return len(self.curr_loader.dataset) 