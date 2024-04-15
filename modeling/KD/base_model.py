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
        self.dataset = self.teacher.dataset
        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items
        self.lmbda = args.lmbda

        if self.frozen_teacher:
            for param in self.teacher.parameters():
                param.requires_grad=False

    def get_loss(self):
        raise NotImplementedError
    
    def get_ratings(self, batch_user):
        return self.student.get_ratings(batch_user)

    def do_something_in_each_epoch(self, epoch):
        return
    
    def train(self):
        self.training = True
        self.student.train()

    def eval(self):
        self.training = False
        self.student.eval()
    
    def get_params_to_update(self):
        return [{"params": self.student.parameters(), 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        output = self.student(batch_user, batch_pos_item, batch_neg_item)
        base_loss = self.student.get_loss(output)
        kd_loss = self.get_loss(batch_user, batch_pos_item, batch_neg_item)
        loss = base_loss + self.lmbda * kd_loss
        return loss

    def save(self):
        return self.student.state_dict(), os.path.join("checkpoints", self.args.dataset, self.args.backbone, f"{self.args.model.lower()}-{self.student.embedding_dim}.pt")
