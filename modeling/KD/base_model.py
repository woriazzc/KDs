import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseKD(nn.Module):
    def __init__(self, args, teacher, student, frozen_teacher=True):
        super().__init__()
        
        self.args = args
        self.training = True
        self.teacher = teacher
        self.student = student
        self.frozen_teacher = frozen_teacher
        self.teacher.eval()
        self.lmbda = args.lmbda

        if self.frozen_teacher:
            for param in self.teacher.parameters():
                param.requires_grad = False

    def get_loss(self):
        raise NotImplementedError
    
    def get_ratings(self, param):
        raise NotImplementedError

    def do_something_in_each_epoch(self, epoch):
        return
    
    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)
    
    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def forward(self, data):
        raise NotImplementedError

    @property
    def param_to_save(self):
        return self.student.state_dict()
    
    @property
    def score_mat_to_save(self):
        with torch.no_grad():
            training = self.student.training
            self.student.eval()
            score_mat = self.student.score_mat().cpu()
            self.student.train(training)
            return score_mat


class BaseKD4Rec(BaseKD):
    def __init__(self, args, teacher, student, frozen_teacher=True):
        super().__init__(args, teacher, student, frozen_teacher)
        self.dataset = self.student.dataset
        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        output = self.student(batch_user, batch_pos_item, batch_neg_item)
        base_loss = self.student.get_loss(output)
        kd_loss = self.get_loss(batch_user, batch_pos_item, batch_neg_item)
        loss = base_loss + self.lmbda * kd_loss
        return loss, base_loss.detach(), kd_loss.detach()
    
    def get_ratings(self, batch_user):
        return self.student.get_ratings(batch_user)


class BaseKD4CTR(BaseKD):
    def __init__(self, args, teacher, student, frozen_teacher=True):
        super().__init__(args, teacher, student, frozen_teacher)
    
    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': 0}]

    def forward(self, data, label):
        output = self.student(data)
        base_loss = self.student.get_loss(output, label)
        kd_loss = self.get_loss(data, label)
        loss = base_loss + self.lmbda * kd_loss
        return loss, base_loss.detach(), kd_loss.detach()

    def get_ratings(self, data):
        return self.student(data)
