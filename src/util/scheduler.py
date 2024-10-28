"""
LR scheduler with warmup
"""
import torch
import math

class CosineAnealingLRSchedulerWithWarmup():
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        total_epochs,
        warmup_lr,
        final_lr,
        warmup_start_lr=0,
        warmup_end_lr=0,
        warmup_end_epoch=0
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.final_lr = final_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = warmup_end_lr
        self.warmup_end_epoch = warmup_end_epoch
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            self._warmup_step()
        else:
            self._cosine_annealing_step()
        self.current_epoch += 1
    
    def _warmup_step(self):
        if self.warmup_end_epoch == 0:
            lr = self.warmup_start_lr + (self.warmup_lr - self.warmup_start_lr) * self.current_epoch / self.warmup_epochs
        else:
            lr = self.warmup_start_lr + (self.warmup_end_lr - self.warmup_start_lr) * self.current_epoch / self.warmup_end_epoch
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _cosine_annealing_step(self):
        lr = self.final_lr + 0.5 * (self.warmup_lr - self.final_lr) * (1 + math.cos(math.pi * (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

class CosineAnealingWDScheduler():
    def __init__(
        self,
        optimizer,
        total_epochs,
        init_weight_decay,
        final_weight_decay=0,
    ):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.init_weight_decay = init_weight_decay
        self.final_weight_decay = final_weight_decay
        self.current_epoch = 0
        
    def step(self):
        wd = self.init_weight_decay + (self.final_weight_decay - self.init_weight_decay) * (1 + math.cos(math.pi * self.current_epoch / self.total_epochs)) / 2
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = wd
        self.current_epoch += 1
    
    def get_wd(self):
        return self.optimizer.param_groups[0]['weight_decay']
