import os
import re
import math
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import Projector
from ..base_model import BaseKD4MM
from ..cf import RRD


class FitNet(BaseKD4MM):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "fitnet"
        self.norm = args.fitnet_norm
        self.dropout_rate = args.dropout_rate
        self.hidden_dim_ratio = args.hidden_dim_ratio
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim
        self.projector_u = Projector(self.student_dim, self.teacher_dim, 1, norm=False, dropout_rate=self.dropout_rate, hidden_dim_ratio=self.hidden_dim_ratio)
        self.projector_i = Projector(self.student_dim, self.teacher_dim, 1, norm=False, dropout_rate=self.dropout_rate, hidden_dim_ratio=self.hidden_dim_ratio)

    def get_features(self, batch_entity, is_user):
        if is_user:
            s = self.student.get_user_embedding(batch_entity)
            t = self.teacher.get_user_embedding(batch_entity)
            s_proj = self.projector_u(s)
            if self.norm:
                s_proj = F.normalize(s_proj, p=2, dim=-1)
                t = F.normalize(t, p=2, dim=-1)
        else:
            s = self.student.get_item_embedding(batch_entity)
            t = self.teacher.get_item_embedding(batch_entity)
            s_proj = self.projector_i(s)
            if self.norm:
                s_proj = F.normalize(s_proj, p=2, dim=-1)
                t = F.normalize(t, p=2, dim=-1)
        return t, s_proj
    
    def get_DE_loss(self, batch_entity, is_user):
        T_feas, S_feas = self.get_features(batch_entity, is_user)
        G_diff = (T_feas - S_feas).pow(2).sum(-1)
        loss = G_diff.sum()
        return loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        batch_item = torch.cat([batch_pos_item.unique(), batch_neg_item.unique()]).unique()
        loss_user = self.get_DE_loss(batch_user, True)
        loss_item = self.get_DE_loss(batch_item, False)
        loss = loss_user + loss_item
        return loss
