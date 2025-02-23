import os
import re
import math
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from ..utils import Expert, CKA, info_abundance, Projector, log_rsd, log_norm, save_embedding, log_mmd
from ..base_model import BaseKD4CTR


class NoKD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "nokd"
    
    def get_loss(self, data, label):
        return torch.tensor(0.).cuda()


class WarmUp(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "warmup"
        self.freeze = args.warmup_freeze
        self.student.embedding_layer = deepcopy(self.teacher.embedding_layer)
        for param in self.student.embedding_layer.parameters():
                param.requires_grad = (not self.freeze)
    
    def get_loss(self, data, label):
        return torch.tensor(0.).cuda()


class FitNet(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "fitnet"
        self.layer = args.fitnet_layer
        self.verbose = args.verbose
        if self.layer == "embedding":
            self.projector = nn.Linear(self.student.embedding_layer_dim, self.teacher.embedding_layer_dim)
        elif self.layer == "penultimate":
            if isinstance(self.teacher._penultimate_dim, int):
                # For one-stream models
                teacher_penultimate_dim = self.teacher._penultimate_dim
                student_penultimate_dim = self.student._penultimate_dim
                self.projector = Projector(student_penultimate_dim, teacher_penultimate_dim, 1, norm=False, dropout_rate=0., shallow=True)
            else:
                # For two-stream models
                cross_dim, deep_dim = self.teacher._penultimate_dim
                student_penultimate_dim = self.student._penultimate_dim
                self.projector_cross = nn.Linear(student_penultimate_dim, cross_dim)
                self.projector_deep = nn.Linear(student_penultimate_dim, deep_dim)
        elif self.layer == "none":
            pass
        else:
            raise ValueError
        
    # def do_something_in_each_epoch(self, epoch):
    #     self.epoch = epoch
    #     if self.verbose:
    #         if epoch > 0 and not self.cka is None:
    #             print(self.cka)
    #         self.cka = None
    #         self.cnt = 0
    
    def get_loss(self, data, label):
        if self.layer == "embedding":
            S_emb = self.student.forward_embed(data)
            S_emb = S_emb.reshape(S_emb.shape[0], -1)
            T_emb = self.teacher.forward_embed(data)
            T_emb = T_emb.reshape(T_emb.shape[0], -1)
            S_emb_proj = self.projector(S_emb)
            loss = (T_emb.detach() - S_emb_proj).pow(2).sum(-1).mean()
        elif self.layer == "penultimate":
            S_emb = self.student.forward_penultimate(data)
            T_emb = self.teacher.forward_penultimate(data)
            if isinstance(self.teacher._penultimate_dim, int):
                S_emb_proj = self.projector(S_emb)
                loss = (T_emb.detach() - S_emb_proj).pow(2).sum(-1).mean()
            else:
                S_emb_cross = self.projector_cross(S_emb)
                S_emb_deep = self.projector_deep(S_emb)
                T_emb_cross, T_emb_deep = T_emb
                loss = (T_emb_cross.detach() - S_emb_cross).pow(2).sum(-1).mean() * 0.5 + (T_emb_deep.detach() - S_emb_deep).pow(2).sum(-1).mean() * 0.5
        elif self.layer == "none":
            loss = torch.tensor(0.).cuda()
        else: raise ValueError

        # if self.verbose and self.cnt < 5:
        #     # # calculate CKA
        #     # with torch.no_grad():
        #     #     S_embs = self.student.forward_all_feature(data)
        #     #     T_embs = self.teacher.forward_all_feature(data)
        #     #     CKA_mat = np.zeros((len(T_embs), len(S_embs)))
        #     #     for id_T, T_emb in enumerate(T_embs):
        #     #         for id_S, S_emb in enumerate(S_embs):
        #     #             CKA_mat[id_T, id_S] = CKA(T_emb, S_emb).item()
        #     #     if self.cka is None:
        #     #         self.cka = CKA_mat
        #     #     else:
        #     #         self.cka = (self.cka * self.cnt + CKA_mat) / (self.cnt + 1)
        #     #         self.cnt += 1

        #     # calculate information abundance
        #     with torch.no_grad():
        #         info_S = info_abundance(S_emb)
        #         info_T = info_abundance(T_emb)
        #         info_S_proj = info_abundance(S_emb_proj)
        #         print(f"infoS:{info_S:.2f}, infoT:{info_T:.2f}, infoS_proj:{info_S_proj:.2f}")
        #         self.cnt += 1
        return loss


class RKD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "rkd"
        self.K = args.rkd_K

    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)[:self.K]
        T_emb = self.teacher.forward_penultimate(data)[:self.K]
        S_mat = S_emb.mm(S_emb.T)
        T_mat = T_emb.mm(T_emb.T)
        return (S_mat - T_mat).pow(2).mean()


class BCED(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "bced"

    def get_loss(self, feature, label):
        logit_S = self.student(feature)
        logit_T = self.teacher(feature)
        y_T = torch.sigmoid(logit_T)
        loss = F.binary_cross_entropy_with_logits(logit_S, y_T.float())
        return loss


class CLID(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "clid"

    def get_loss(self, feature, label):
        logit_S = self.student(feature).squeeze(1)
        logit_T = self.teacher(feature).squeeze(1)
        y_T = torch.sigmoid(logit_T)
        y_S = torch.sigmoid(logit_S)
        y_T = y_T / y_T.sum()
        y_S = y_S / y_S.sum()
        loss = F.binary_cross_entropy(y_S, y_T)
        return loss


class OFA(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "ofa"
        self.beta = args.ofa_beta
        self.layer_dims = self.student._all_layer_dims
        self.projectors = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        ) for dim in self.layer_dims])

    def get_loss(self, feature, label):
        logit_S = self.student(feature)
        logit_T = self.teacher(feature)
        y_T = torch.sigmoid(logit_T)
        loss_kd = F.binary_cross_entropy_with_logits(logit_S, y_T)
        loss_ofa = 0.
        features = self.student.forward_all_feature(feature)
        for idx, h in enumerate(features):
            logit_h = self.projectors[idx](h)
            loss_ofa += F.binary_cross_entropy_with_logits(logit_h, y_T)
        loss = loss_kd + loss_ofa * self.beta
        return loss
