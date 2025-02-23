import os
import math
import mlflow
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import Projector, pca, load_pkls, dump_pkls, self_loop_graph, CKA, info_abundance, log_mmd, TCA, MMD_loss
from ..base_model import BaseKD4CTR


class HetD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "hetd"
        self.beta = args.hetd_beta
        self.gamma = args.hetd_gamma
        self.lmbda = args.lmbda
        self.verbose = args.verbose

        student_penultimate_dim = self.student._penultimate_dim
        if isinstance(self.teacher._penultimate_dim, int):
            # For one-stream models
            teacher_penultimate_dim = self.teacher._penultimate_dim
        else:
            # For two-stream models
            cross_dim, deep_dim = self.teacher._penultimate_dim
            teacher_penultimate_dim = cross_dim + deep_dim
        self.projector = nn.Linear(student_penultimate_dim, teacher_penultimate_dim)
        self.adaptor = nn.Linear(teacher_penultimate_dim, teacher_penultimate_dim)
        self.predictor = nn.Linear(teacher_penultimate_dim, 1)
    
    # def do_something_in_each_epoch(self, epoch):
    #     if self.verbose:
    #         if epoch > 0 and not self.cka is None:
    #             print(self.cka)
    #         self.cka = None
    #         self.cnt = 0
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        if isinstance(self.teacher._penultimate_dim, tuple):
            # Two-stream models
            T_emb_cross, T_emb_deep = T_emb
            T_emb = torch.cat([T_emb_cross, T_emb_deep], dim=-1)
        S_emb = self.projector(S_emb)
        T_emb = self.adaptor(T_emb)
        T_logits = self.predictor(T_emb)
        S_pred = torch.sigmoid(self.student(data))
        loss_adaptor = F.binary_cross_entropy_with_logits(T_logits.squeeze(-1), label.squeeze(-1).float()) + self.beta * F.binary_cross_entropy_with_logits(T_logits.squeeze(-1), S_pred.detach().squeeze(-1))
        loss = (T_emb.detach() - S_emb).pow(2).sum(-1).mean() + loss_adaptor / (self.lmbda + 1e-8) * self.gamma

        # if self.verbose and self.cnt < 5:
        #     # # calculate CKA
        #     # with torch.no_grad():
        #     #     S_embs = self.student.forward_all_feature(data)
        #     #     T_embs = self.teacher.forward_all_feature(data)
        #     #     T_embs += [T_emb.detach()]
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
        #         S_emb = self.student.forward_penultimate(data)
        #         T_emb = self.teacher.forward_penultimate(data)
        #         S_emb = S_emb.reshape(S_emb.shape[0], -1)
        #         T_emb = T_emb.reshape(T_emb.shape[0], -1)
        #         info_S = info_abundance(S_emb)
        #         info_T = info_abundance(T_emb)
        #         print(info_S, info_T, end=" ")
        #         S_emb = self.projector(S_emb)
        #         T_emb = self.adaptor(T_emb)
        #         info_S = info_abundance(S_emb)
        #         info_T = info_abundance(T_emb)
        #         print(info_S, info_T)
        #         self.cnt += 1
        
        return loss


class PairD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "paird"
        self.beta = args.paird_beta
        self.tau = args.paird_tau

    def do_something_in_each_epoch(self, epoch):
        if epoch != 0: print(self.grad1, self.grad2)
        self.cnt = 0
        self.grad1, self.grad2 = 0., 0.

    def get_loss(self, feature, label):
        logit_S = self.student(feature).squeeze(1)
        logit_T = self.teacher(feature).squeeze(1)

        # randidx = torch.arange(len(logit_T)).flip(-1)
        randidx = torch.randperm(len(logit_T))
        neg_T, pos_T = logit_T.clone(), logit_T[randidx].clone()
        idx = torch.argwhere(neg_T > pos_T)
        neg_T[idx], pos_T[idx] = pos_T[idx], neg_T[idx]
        neg_S, pos_S = logit_S.clone(), logit_S[randidx].clone()
        neg_S[idx], pos_S[idx] = pos_S[idx], neg_S[idx]
        gap_T = pos_T - neg_T
        gap_S = pos_S.detach() - neg_S
        idx = torch.argwhere(neg_S < torch.quantile(logit_S, 0.01).detach())
        y_T = torch.sigmoid(gap_T[idx] / self.tau)
        y_S = torch.sigmoid(gap_S[idx] / self.tau)

        loss_rk = F.binary_cross_entropy(y_S, y_T)
        y_T = torch.sigmoid(logit_T)
        loss_bce = F.binary_cross_entropy_with_logits(logit_S, y_T)
        loss = self.beta * loss_rk + (1. - self.beta) * loss_bce
        
        with torch.no_grad():
            # grad1 = (torch.sigmoid(logit_S) - torch.sigmoid(logit_T)).mean().detach().cpu().item()
            # grad2 = (torch.sigmoid(gap_T) - torch.sigmoid(gap_S)).mean().detach().cpu().item()
            grad1 = torch.autograd.grad(loss_bce, logit_S, retain_graph=True)[0].sum().detach().cpu().item()
            grad2 = torch.autograd.grad(loss_rk, logit_S, retain_graph=True)[0].sum().detach().cpu().item()
            self.grad1 = (self.grad1 * self.cnt + grad1) / (self.cnt + 1)
            self.grad2 = (self.grad2 * self.cnt + grad2) / (self.cnt + 1)
        return loss


class FFFit(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "fffit"
        self.num_fields = self.student.num_fields
        self.projectors = nn.ModuleList([nn.Linear(self.student.embedding_dim, self.teacher.embedding_dim) for _ in range(self.num_fields)])
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_embed(data)    # bs, num_fields, embedding_dim
        T_emb = self.teacher.forward_embed(data)
        loss = 0.
        for field in range(self.num_fields):
            projector = self.projectors[field]
            T_field_emb = T_emb[:, field, :]
            S_field_emb = projector(S_emb[:, field, :])
            loss += (T_field_emb.detach() - S_field_emb).pow(2).sum(-1).mean()
        return loss


class AnyD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "anyd"
        self.T_layer = args.T_layer
        self.S_layer = args.S_layer
        self.T_dim = self.teacher.get_layer_dim(self.T_layer)
        self.S_dim = self.student.get_layer_dim(self.S_layer)
        self.projector = nn.Linear(self.S_dim, self.T_dim)
        nn.init.xavier_normal_(self.projector.weight)
        nn.init.constant_(self.projector.bias, 0)

    def get_loss(self, data, label):
        S_emb = self.student.forward_layer(data, self.S_layer)    # bs, layer_dim
        T_emb = self.teacher.forward_layer(data, self.T_layer)
        S_emb = self.projector(S_emb)
        loss = (T_emb.detach() - S_emb).pow(2).sum(-1).mean()
        return loss


class adaD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "adad"
        self.beta = args.adad_beta
        self.gamma = args.adad_gamma
        self.lmbda = args.lmbda
        self.verbose = args.verbose

        student_penultimate_dim = self.student._penultimate_dim
        if isinstance(self.teacher._penultimate_dim, int):
            # For one-stream models
            teacher_penultimate_dim = self.teacher._penultimate_dim
        else:
            # For two-stream models
            cross_dim, deep_dim = self.teacher._penultimate_dim
            teacher_penultimate_dim = cross_dim + deep_dim
        self.projector = nn.Linear(student_penultimate_dim, teacher_penultimate_dim)
        self.adaptor_S = nn.Linear(student_penultimate_dim, student_penultimate_dim)
        self.adaptor_T = nn.Linear(teacher_penultimate_dim, teacher_penultimate_dim)
        self.predictor_S = nn.Linear(student_penultimate_dim, 1)
        self.predictor_T = nn.Linear(teacher_penultimate_dim, 1)
    
    def do_something_in_each_epoch(self, epoch):
        # for embedding in self.student.embedding_layer.embedding.items():
        #     print(int(info_abundance(embedding[1].weight.data)), end=" ")
        # print()
        # for embedding in self.teacher.embedding_layer.embedding.items():
        #     print(int(info_abundance(embedding[1].weight.data)), end=" ")
        if self.verbose:
            if epoch > 0 and not self.cka is None:
                print(self.cka)
            self.cka = None
            self.cnt = 0
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        if isinstance(self.teacher._penultimate_dim, tuple):
            # Two-stream models
            T_emb_cross, T_emb_deep = T_emb
            T_emb = torch.cat([T_emb_cross, T_emb_deep], dim=-1)
        T_emb_adapt = self.adaptor_T(T_emb)
        S_emb_adapt = self.adaptor_S(S_emb.detach())
        T_logits_adapt = self.predictor_T(T_emb_adapt)
        S_logits_adapt = self.predictor_S(S_emb_adapt)
        T_pred = torch.sigmoid(self.teacher(data))
        S_pred = torch.sigmoid(self.student(data))
        loss_adapt_T = F.binary_cross_entropy_with_logits(T_logits_adapt.squeeze(-1), label.squeeze(-1).float()) + self.beta * F.binary_cross_entropy_with_logits(T_logits_adapt.squeeze(-1), S_pred.detach().squeeze(-1))
        loss_adapt_S = F.binary_cross_entropy_with_logits(S_logits_adapt.squeeze(-1), label.squeeze(-1).float()) + self.beta * F.binary_cross_entropy_with_logits(S_logits_adapt.squeeze(-1), T_pred.detach().squeeze(-1))
        adaptor_S_detach = deepcopy(self.adaptor_S)
        for param in adaptor_S_detach.parameters():
                param.requires_grad = False
        S_emb_proj = self.projector(adaptor_S_detach(S_emb))
        loss = (T_emb_adapt.detach() - S_emb_proj).pow(2).sum(-1).mean() + (loss_adapt_S + loss_adapt_T) / (self.lmbda + 1e-8) * self.gamma
        
        if self.verbose and self.cnt < 5:
            # # calculate CKA
            # with torch.no_grad():
            #     S_embs = self.student.forward_all_feature(data)
            #     T_embs = self.teacher.forward_all_feature(data)
            #     T_embs += [T_emb.detach()]
            #     CKA_mat = np.zeros((len(T_embs), len(S_embs)))
            #     for id_T, T_emb in enumerate(T_embs):
            #         for id_S, S_emb in enumerate(S_embs):
            #             CKA_mat[id_T, id_S] = CKA(T_emb, S_emb).item()
            #     if self.cka is None:
            #         self.cka = CKA_mat
            #     else:
            #         self.cka = (self.cka * self.cnt + CKA_mat) / (self.cnt + 1)
            #         self.cnt += 1

            # calculate information abundance
            with torch.no_grad():
                info_S = info_abundance(S_emb)
                info_T = info_abundance(T_emb)
                info_adapt_S = info_abundance(S_emb_adapt)
                info_adapt_T = info_abundance(T_emb_adapt)
                info_proj_S = info_abundance(S_emb_proj)
                print(info_S, info_T, info_adapt_S, info_adapt_T, info_proj_S)
                self.cnt += 1

            # calculate information abundance for each field
            # with torch.no_grad():
            #     # for i in range(self.student.num_fields):
            #     #     emb = S_emb[:, i*self.student.embedding_dim:(i+1)*self.student.embedding_dim]
            #     #     print(int(info_abundance(emb)), end=" ")
            #     # print()
            #     for i in range(self.teacher.num_fields):
            #         emb = T_emb[:, i*self.teacher.embedding_dim:(i+1)*self.teacher.embedding_dim]
            #         print(int(info_abundance(emb)), end=" ")
            #     print()
            #     self.cnt += 1
        
        return loss


class copyD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "copyd"
        self.num_experts = args.num_experts
        self.beta = args.copyd_beta
        self.gamma = args.copyd_gamma
        self.teacher_linear = self.teacher.linear
        self.projector = Projector(self.student._penultimate_dim, self.teacher._penultimate_dim, self.num_experts, norm=False, dropout_rate=0., shallow=False)
    
    # def get_ratings(self, data):
    #     old_linear = deepcopy(self.student.linear)
    #     self.student.linear = nn.Sequential(self.projector, self.teacher_linear)
    #     res = self.student(data)
    #     self.student.linear = old_linear
    #     return res
        
    def do_something_in_each_epoch(self, epoch):
        self.epoch = epoch
        if epoch != 0: print(f"loss_emb:{self.loss_emb}, loss_linear:{self.loss_linear}")
        self.loss_emb, self.loss_linear, self.cnt = 0, 0, 0
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        S_emb_proj = self.projector(S_emb)
        loss_emb = (T_emb.detach() - S_emb_proj).pow(2).sum(-1).mean()
        # S_logits = self.student.linear(S_emb.detach())
        # proj_linear_logits = self.teacher_linear(S_emb_proj).detach()
        # y_T = torch.sigmoid(proj_linear_logits)
        # loss_linear = F.binary_cross_entropy_with_logits(S_logits, y_T.float())
        # loss_linear = F.binary_cross_entropy_with_logits(S_logits.squeeze(-1), label.squeeze(-1).float())
        # for _, module in self.student.linear.named_modules():
        #         for p_name, param in module.named_parameters():
        #             if param.requires_grad:
        #                 if p_name in ["weight", "bias"]:
        #                     loss_linear += torch.norm(param, p=2) * self.student.L2_weight
        loss = loss_emb
        self.loss_emb = (self.loss_emb * self.cnt + loss_emb.detach().item()) / (self.cnt + 1)
        # self.loss_linear = (self.loss_linear * self.cnt + loss_linear.detach().item()) / (self.cnt + 1)
        self.cnt += 1
        return loss

    def forward(self, data, label):
        # self.student.linear = nn.Sequential(self.projector, self.teacher_linear)
        output = self.student(data)
        base_loss = self.student.get_loss(output, label)
        kd_loss = self.get_loss(data, label)
        loss = kd_loss + self.lmbda * base_loss
        return loss, base_loss.detach(), kd_loss.detach()


class fieldD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "fieldd"
        self.num_experts = args.num_experts
        self.beta = args.fieldd_beta
        self.gamma = args.fieldd_gamma
        self.teacher_linear = self.teacher.linear
        self.projectors = nn.ModuleList([Projector(self.student._penultimate_dim, self.teacher._penultimate_dim // self.student.num_fields, 1, norm=False, dropout_rate=0., shallow=True) for _ in range(self.student.num_fields)])

    # def get_ratings(self, data):
    #     old_linear = deepcopy(self.student.linear)
    #     self.student.linear = nn.Sequential(self.projector, self.teacher_linear)
    #     res = self.student(data)
    #     self.student.linear = old_linear
    #     return res
    
    def do_something_in_each_epoch(self, epoch):
        self.epoch = epoch
        if epoch != 0: print(f"loss_emb:{self.loss_emb}, loss_linear:{self.loss_linear}")
        self.loss_emb, self.loss_linear, self.cnt = 0, 0, 0
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        S_emb_proj = []
        for i in range(self.teacher.num_fields):
            S_emb_proj.append(self.projectors[i](S_emb))
        S_emb_proj = torch.cat(S_emb_proj, dim=-1)
        loss_emb = (T_emb.detach() - S_emb_proj).pow(2).sum(-1).mean()
        S_logits = self.student.linear(S_emb.detach())
        proj_linear_logits = self.teacher_linear(S_emb_proj).detach()
        y_T = torch.sigmoid(proj_linear_logits)
        loss_linear = F.binary_cross_entropy_with_logits(S_logits, y_T.float())
        loss_linear = self.gamma * loss_linear + (1. - self.gamma) * F.binary_cross_entropy_with_logits(S_logits.squeeze(-1), label.squeeze(-1).float())
        loss = self.beta * loss_emb + (1. - self.beta) * loss_linear
        self.loss_emb = (self.loss_emb * self.cnt + loss_emb.detach().item()) / (self.cnt + 1)
        self.loss_linear = (self.loss_linear * self.cnt + loss_linear.detach().item()) / (self.cnt + 1)
        self.cnt += 1
        return loss

    def forward(self, data, label):
        output = self.student(data)
        base_loss = self.student.get_loss(output, label)
        kd_loss = self.get_loss(data, label)
        loss = kd_loss
        return loss, base_loss.detach(), kd_loss.detach()


class watD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "watd"
        self.beta = args.watd_beta
        self.adaptor = Projector(self.teacher._penultimate_dim, self.student._penultimate_dim, 1, norm=False, dropout_rate=0., shallow=True)
        self.predictor = nn.Linear(self.student._penultimate_dim, 1)
    
    def do_something_in_each_epoch(self, epoch):
        self.epoch = epoch
        if epoch != 0: print(f"loss_emb:{self.loss_emb}, loss_adapt:{self.loss_adapt}")
        self.loss_emb, self.loss_adapt, self.cnt = 0, 0, 0
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data).detach()
        T_emb_adapt = self.adaptor(T_emb)
        T_logit = self.teacher(data)
        T_adapt_logit = self.predictor(T_emb_adapt)
        loss_emb = (T_emb_adapt.detach() - S_emb).pow(2).sum(-1).mean()
        T_pred = torch.sigmoid(T_logit)
        loss_adapt = F.binary_cross_entropy_with_logits(T_adapt_logit, T_pred)
        loss = loss_emb + self.beta * loss_adapt
        self.loss_emb = (self.loss_emb * self.cnt + loss_emb.detach().item()) / (self.cnt + 1)
        self.loss_adapt = (self.loss_adapt * self.cnt + loss_adapt.detach().item()) / (self.cnt + 1)
        self.cnt += 1
        return loss

    def forward(self, data, label):
        output = self.student(data)
        base_loss = self.student.get_loss(output, label)
        kd_loss = self.get_loss(data, label)
        loss = kd_loss
        return loss, base_loss.detach(), kd_loss.detach()


class attachD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "attachd"
        self.num_experts = args.num_experts
        self.teacher_linear = self.teacher.linear
        self.projector = Projector(self.student._penultimate_dim, self.teacher._penultimate_dim, self.num_experts, norm=False, dropout_rate=0., shallow=False)
        self.student.linear = Projector(self.student._penultimate_dim, 1, 1, norm=False, dropout_rate=0., shallow=False)
    
    # def get_ratings(self, data):
    #     old_linear = deepcopy(self.student.linear)
    #     self.student.linear = nn.Sequential(self.projector, self.teacher_linear)
    #     res = self.student(data)
    #     self.student.linear = old_linear
    #     return res
        
    def do_something_in_each_epoch(self, epoch):
        self.epoch = epoch
        if epoch != 0: print(f"loss_emb:{self.loss_emb}, loss_linear:{self.loss_linear}")
        self.loss_emb, self.loss_linear, self.cnt = 0, 0, 0
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        S_emb_proj = self.projector(S_emb)
        loss_emb = (T_emb.detach() - S_emb_proj).pow(2).sum(-1).mean()
        S_logits = self.student.linear(S_emb.detach())
        proj_linear_logits = self.teacher_linear(S_emb_proj).detach()
        y_T = torch.sigmoid(proj_linear_logits)
        loss_linear = F.binary_cross_entropy_with_logits(S_logits, y_T.float())
        loss = loss_emb + loss_linear
        self.loss_emb = (self.loss_emb * self.cnt + loss_emb.detach().item()) / (self.cnt + 1)
        self.loss_linear = (self.loss_linear * self.cnt + loss_linear.detach().item()) / (self.cnt + 1)
        self.cnt += 1
        return loss

    def forward(self, data, label):
        output = self.student(data)
        base_loss = self.student.get_loss(output, label)
        kd_loss = self.get_loss(data, label)
        loss = kd_loss
        return loss, base_loss.detach(), kd_loss.detach()


class TCAD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "tcad"
        self.layer = args.tcad_layer
        self.lmbda = args.tcad_lmbda
        self.K = args.tcad_K
        self.verbose = args.verbose
        self.ablation = args.ablation
        if self.layer == "embedding":
            self.tca = TCA(self.student.embedding_layer_dim, self.lmbda, self.K)
            self.projector = Projector(self.student.embedding_layer_dim, self.teacher.embedding_layer_dim, 1, norm=False, dropout_rate=0., shallow=True)
        elif self.layer == "penultimate":
            self.tca = TCA(self.student._penultimate_dim, self.lmbda, self.K, False)
            self.projector = Projector(self.student._penultimate_dim, self.teacher._penultimate_dim, 1, norm=False, dropout_rate=0., shallow=True)
        elif self.layer == "none":
            pass
        else:
            raise ValueError
        self.step = 0
    
    def get_loss(self, data, label):
        if self.layer == "embedding":
            S_emb = self.student.forward_embed(data)
            S_emb = S_emb.reshape(S_emb.shape[0], -1)
            T_emb = self.teacher.forward_embed(data)
            T_emb = T_emb.reshape(T_emb.shape[0], -1)
        elif self.layer == "penultimate":
            S_emb = self.student.forward_penultimate(data)
            T_emb = self.teacher.forward_penultimate(data)
        if self.layer in ["embedding", "penultimate"]:
            if self.ablation:
                S_emb_adapt = S_emb / torch.linalg.norm(S_emb, dim=-1, keepdim=True).detach()
                T_emb_adapt = T_emb / torch.linalg.norm(T_emb, dim=-1, keepdim=True).detach()
            else:
                if self.step % 500 == 0:
                    # S_emb_pos, S_emb_neg = S_emb[label == 1], S_emb[label == 0]
                    # T_emb_pos, T_emb_neg = T_emb[label == 1], T_emb[label == 0]
                    # S_emb_pos_adapt, T_emb_pos_adapt = self.tca.fit_transform(S_emb_pos, T_emb_pos)
                    # S_emb_neg_adapt, T_emb_neg_adapt = self.tca.fit_transform(S_emb_neg, T_emb_neg)
                    S_emb_adapt, T_emb_adapt = self.tca.fit_transform(S_emb, T_emb, label)
                    loss = (T_emb_adapt.detach() - S_emb_adapt).pow(2).sum(-1).mean()
                    mmd = MMD_loss()(S_emb_adapt[-512:], T_emb_adapt[-512:])
                    # mmd_pos = MMD_loss()(S_emb_pos_adapt[-512:], T_emb_pos_adapt[-512:])
                    # mmd_neg = MMD_loss()(S_emb_neg_adapt[-512:], T_emb_neg_adapt[-512:])
                    if self.verbose:
                        mlflow.log_metrics({"mmd":mmd.item()}, step=self.step)
                        mlflow.log_metrics({"info_S_emb":info_abundance(S_emb), "info_T_emb":info_abundance(T_emb), "info_S_adapt":info_abundance(S_emb_adapt), "info_T_adapt":info_abundance(T_emb_adapt)}, step=self.step)
                        # mlflow.log_metrics({"info_S_emb_pos":info_abundance(S_emb_pos), "info_T_emb_pos":info_abundance(T_emb_pos), "info_S_pos_adapt":info_abundance(S_emb_pos_adapt), "info_T_pos_adapt":info_abundance(T_emb_pos_adapt)}, step=self.step)
                        # mlflow.log_metrics({"info_S_emb_neg":info_abundance(S_emb_neg), "info_T_emb_neg":info_abundance(T_emb_neg), "info_S_neg_adapt":info_abundance(S_emb_neg_adapt), "info_T_neg_adapt":info_abundance(T_emb_neg_adapt)}, step=self.step)
                else:
                    loss = torch.tensor(0.).cuda()
                self.step += 1
        elif self.layer == "none":
            loss = torch.tensor(0.).cuda()
        else: raise ValueError
        return loss


class DAND(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "dand"
        self.layer = args.dand_layer
        self.K = args.dand_K
        self.kernel_type = args.kernel_type
        self.ablation = args.ablation
        if self.layer == "embedding":
            self.mmd = MMD_loss(self.kernel_type)
            self.projector = Projector(self.student.embedding_layer_dim, self.teacher.embedding_layer_dim, 1, norm=False, dropout_rate=0., shallow=True)
        elif self.layer == "penultimate":
            self.mmd = MMD_loss(self.kernel_type)
            self.projector = Projector(self.student._penultimate_dim, self.teacher._penultimate_dim, 1, norm=False, dropout_rate=0., shallow=True)
        elif self.layer == "none":
            pass
        else:
            raise ValueError
    
    def get_loss(self, data, label):
        if self.layer == "embedding":
            S_emb = self.student.forward_embed(data)
            S_emb = S_emb.reshape(S_emb.shape[0], -1)
            T_emb = self.teacher.forward_embed(data)
            T_emb = T_emb.reshape(T_emb.shape[0], -1)
        elif self.layer == "penultimate":
            S_emb = self.student.forward_penultimate(data)
            T_emb = self.teacher.forward_penultimate(data)
        if self.layer in ["embedding", "penultimate"]:
            S_emb_proj = self.projector(S_emb)
            if self.ablation:
                loss = (T_emb - S_emb_proj).pow(2).sum(-1).mean()
            else:
                if self.kernel_type == "linear":
                    loss = self.mmd(S_emb_proj, T_emb)
                else:
                    loss = self.mmd(S_emb_proj[:self.K], T_emb[:self.K])
        elif self.layer == "none":
            loss = torch.tensor(0.).cuda()
        else: raise ValueError
        return loss


class MixD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "mixd"
        self.layer = args.mixd_layer
        self.K = args.mixd_K
        self.kernel_type = args.kernel_type
        self.verbose = args.verbose
        self.ablation = args.ablation
        if self.layer == "embedding":
            self.mmd = MMD_loss(self.kernel_type)
            self.projector_pos = Projector(self.student.embedding_layer_dim, self.teacher.embedding_layer_dim, 1, norm=False, dropout_rate=0., shallow=True)
            self.projector_neg = Projector(self.student.embedding_layer_dim, self.teacher.embedding_layer_dim, 1, norm=False, dropout_rate=0., shallow=True)
        elif self.layer == "penultimate":
            self.mmd = MMD_loss(self.kernel_type)
            self.projector_pos = Projector(self.student._penultimate_dim, self.teacher._penultimate_dim, 1, norm=False, dropout_rate=0., shallow=True)
            self.projector_neg = Projector(self.student._penultimate_dim, self.teacher._penultimate_dim, 1, norm=False, dropout_rate=0., shallow=True)
        elif self.layer == "none":
            pass
        else:
            raise ValueError
    
    def _cal_loss(self, S_emb, T_emb, projector):
        S_emb_proj = projector(S_emb)
        if self.ablation:
            loss = (T_emb[:self.K] - S_emb_proj[:self.K]).pow(2).sum(-1).mean()
            return loss
        if self.kernel_type == "linear":
            loss = self.mmd(S_emb_proj[:self.K], T_emb[:self.K])
        else:
            loss = self.mmd(S_emb_proj[:self.K], T_emb[:self.K])
        return loss
    
    def get_loss(self, data, label):
        if self.layer == "embedding":
            S_emb = self.student.forward_embed(data)
            S_emb = S_emb.reshape(S_emb.shape[0], -1)
            T_emb = self.teacher.forward_embed(data)
            T_emb = T_emb.reshape(T_emb.shape[0], -1)
        elif self.layer == "penultimate":
            S_emb = self.student.forward_penultimate(data)
            T_emb = self.teacher.forward_penultimate(data)
        if self.layer in ["embedding", "penultimate"]:
            idx_pos = torch.argwhere(label == 1).squeeze(-1)
            idx_neg = torch.argwhere(label == 0).squeeze(-1)
            S_emb_pos, S_emb_neg = S_emb[idx_pos], S_emb[idx_neg]
            T_emb_pos, T_emb_neg = T_emb[idx_pos], T_emb[idx_neg]
            loss_pos = self._cal_loss(S_emb_pos, T_emb_pos, self.projector_pos)
            loss_neg = self._cal_loss(S_emb_neg, T_emb_neg, self.projector_neg)
            loss = (loss_pos + loss_neg) * 0.5
            if self.verbose:
                mlflow.log_metrics({"loss_pos":loss_pos.item(), "loss_neg":loss_neg.item()})
        elif self.layer == "none":
            loss = torch.tensor(0.).cuda()
        else: raise ValueError
        return loss


class RSD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "rsd"
        self.ratio = args.rsd_ratio
    
    def cal_rsd(self, X, Y):
        Us, Ss, Vs = torch.svd(X)
        Ut, St, Vt = torch.svd(Y)
        total_Ss, total_St = torch.sum(Ss), torch.sum(St)
        cumu_Ss, cumu_St = torch.cumsum(Ss, dim=0), torch.cumsum(St, dim=0)
        thresh_s, thresh_t = self.ratio * total_Ss, self.ratio * total_St
        K_s, K_t = torch.where(cumu_Ss >= thresh_s)[0][0] + 1, torch.where(cumu_St >= thresh_t)[0][0] + 1
        Us, Ut = Us[:, :K_s], Ut[:, :K_t]
        Ps, cospa, Pt = torch.svd(torch.mm(Us.T, Ut))
        cospa = torch.minimum(cospa, torch.tensor(1.))
        sinpa = torch.sqrt(1. - torch.pow(cospa, 2))
        rsd = torch.norm(sinpa, 1)
        # bmp = torch.norm(Ps.abs() - Pt.abs(), 2)
        dis = rsd
        return dis
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        loss = self.cal_rsd(S_emb, T_emb)
        return loss


class PCAD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "pcad"
        self.SK = args.pcad_SK
        self.TK = args.pcad_TK
        self.Sr = args.pcad_Sr
        self.Tr = args.pcad_Tr
        self.projector = Projector(self.SK, self.TK, 1, norm=False, dropout_rate=0., shallow=True)

    def pca_with_grad(self, X, K, rat):
        # Xc = X - X.mean(0, keepdims=True)
        U, S, V = torch.svd_lowrank(X, K)
        S_smooth = S.clone()
        S_smooth[:math.ceil(K * rat)] = S[:math.ceil(K * rat)].mean()
        Y = U.mm(torch.diag(S_smooth))
        return Y
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        S_emb = self.pca_with_grad(S_emb, self.SK, self.Sr)
        T_emb = self.pca_with_grad(T_emb, self.TK, self.Tr)
        S_emb_proj = self.projector(S_emb)
        loss = (T_emb - S_emb_proj).pow(2).sum(-1).mean()
        return loss


class CCD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "ccd"
        self.SK = args.ccd_SK
        self.TK = args.ccd_TK
        self.beta = args.ccd_beta
        self.projector = Projector(self.SK, self.TK, 1, norm=False, dropout_rate=0., shallow=True)
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        S_emb = S_emb - S_emb.mean(0, keepdims=True)
        T_emb = T_emb - T_emb.mean(0, keepdims=True)
        Us, Ss, Vs = torch.svd_lowrank(S_emb, self.SK)
        Ut, St, Vt = torch.svd_lowrank(T_emb, self.TK)
        # Us, Ss, Vs = torch.svd(S_emb)
        # Ut, St, Vt = torch.svd(T_emb)
        # Us, Ss, Vs = Us[:, :self.SK], Ss[:self.SK], Vs[:, :self.SK]
        # Ut, St, Vt = Ut[:, :self.TK], St[:self.TK], Vt[:, :self.TK]
        
        # Ys = Us.mm(torch.diag(Ss))     # bs, SK
        # Yt = Ut.mm(torch.diag(St))     # bs, TK
        # loss1 = self.projector(Ys.detach()).pow(2).sum() + Yt.pow(2).sum()
        # # mlflow.log_metric("rk_proj", info_abundance(self.projector.experts[0].mlp[0].weight.data))
        # vec_t = Ut                  # bs, TK
        # # Ss_smooth = Ss.detach().clone()
        # # Ss_smooth[:math.ceil(self.SK * 0.3)] = Ss[:math.ceil(self.SK * 0.3)].mean()
        # # Us = Us * Ss_smooth.unsqueeze(0)   # bs, SK
        # vec_s = self.projector(Us)  # bs, TK

        # vec_t = vec_t / torch.norm(vec_t, dim=0, keepdim=True)
        # vec_s = vec_s / torch.norm(vec_s, dim=0, keepdim=True)
        # cos_mat = torch.diag(vec_t.T.mm(vec_s))     # TK,
        # sin_mat = torch.sqrt(1. - torch.pow(torch.minimum(cos_mat, torch.tensor(1.)), 2))
        # # St_smooth = St.clone()
        # # St_smooth[:math.ceil(self.TK * 0.2)] = St[:math.ceil(self.TK * 0.2)].mean()

        # # d = {f"sin_{i}":sin_mat[i].item() for i in range(10)}
        # # mlflow.log_metrics(d)

        # # St[10:] = 0
        # # St[[1, 2, 6, 7, 8, 9]] = 0
        # # St[1:] = 0

        idx = [0]
        cos_mat = torch.diag(Ut[:, idx].T.mm(Us[:, idx]))
        sin_mat = torch.sqrt(1. - torch.pow(torch.minimum(cos_mat, torch.tensor(1.)), 2))
        mlflow.log_metric("sin_00", sin_mat.mean().item())
        mlflow.log_metric("St", St[idx].mean().item())
        mlflow.log_metric("Ss", Ss[idx].mean().item())
        sin_mat = sin_mat * St[idx] * 10

        # sin_mat = sin_mat * St
        loss2 = 2 * sin_mat.sum()
        # loss = (loss1 * self.beta + loss2) / len(S_emb)
        loss = loss2 / len(S_emb)
        # mlflow.log_metric("loss2", loss2.item())

        # P = self.projector.experts[0].mlp[0].weight.T[:, :10]
        # lam = torch.linalg.svdvals(P)
        # loss3 = -lam.pow(2).sum()
        # loss = (loss1 * self.beta + loss2 + loss3) / len(S_emb)
        # mlflow.log_metric("rk_loss", loss3.item())
        return loss


class conD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "cond"
        self.beta = args.cond_beta
        self.adaptor = Projector(self.teacher._penultimate_dim, self.student._penultimate_dim, 1, norm=False, dropout_rate=0., shallow=False)
        self.projector = Projector(self.student._penultimate_dim, self.student._penultimate_dim, 1, norm=False, dropout_rate=0., shallow=True)

    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        T_adapt_emb = self.adaptor(T_emb)
        predictor = deepcopy(self.student.linear)
        for param in predictor.parameters():
            param.requires_grad = False
        T_logits = predictor(T_adapt_emb)
        loss_T = F.binary_cross_entropy_with_logits(T_logits.squeeze(-1), label.squeeze(-1).float())
        T_adapt_proj = T_adapt_emb
        loss_S = (T_adapt_proj.detach() - S_emb).pow(2).sum(-1).mean()
        loss = loss_T * self.beta + loss_S
        mlflow.log_metrics({"loss_T":loss_T.item(), "loss_S":loss_S.item()})
        return loss


class PDD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "pdd"
        self.K = args.pdd_K
        self.beta = args.pdd_beta
        self.gamma = args.pdd_gamma
        self.adaptor = Projector(self.teacher._penultimate_dim, self.student._penultimate_dim, 1, norm=False, dropout_rate=0.1, shallow=False)
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        T_emb_adapt = self.adaptor(T_emb)
        predictor = deepcopy(self.student.linear)
        for param in predictor.parameters():
            param.requires_grad = False
        T_logits = predictor(T_emb_adapt)
        loss_T = F.binary_cross_entropy_with_logits(T_logits.squeeze(-1), label.squeeze(-1).float())

        Us, Ss, Vs = torch.svd_lowrank(S_emb, self.K)
        Ut, St, Vt = torch.svd_lowrank(T_emb_adapt.detach(), self.K)
        loss1 = Ss.sum()
        cos = torch.diag(Ut.T.mm(Us))
        sin = torch.sqrt(1. - torch.pow(torch.minimum(cos, torch.tensor(1.)), 2))
        loss2 = sin.sum()
        # loss2 = -cos.sum()
        loss = loss_T * self.gamma + loss1 + self.beta * loss2
        return loss


class shrD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "shrd"
        self.K = args.shrd_K
        self.beta = args.shrd_beta
        self.layer = args.shrd_layer
        
    def get_loss(self, data, label):
        if self.layer == "penu":
            S_emb = self.student.forward_penultimate(data)
            T_emb = self.teacher.forward_penultimate(data)
        elif self.layer == "embed":
            S_emb = self.student.forward_embed(data)
            S_emb = S_emb.reshape(S_emb.shape[0], -1)
            T_emb = self.teacher.forward_embed(data)
            T_emb = T_emb.reshape(T_emb.shape[0], -1)
        else: raise ValueError
        S_emb = S_emb - S_emb.mean(0, keepdims=True)
        T_emb = T_emb - T_emb.mean(0, keepdims=True)
        Us, Ss, Vs = torch.svd_lowrank(S_emb, self.K)
        Ut, St, Vt = torch.svd_lowrank(T_emb, self.K)
        # Pt = torch.softmax(St / self.T, dim=0)
        # loss1 = F.cross_entropy(Ss, Pt)
        loss1 = Ss.sum()
        cos = torch.diag(Ut.T.mm(Us))
        sin = torch.sqrt(1. - torch.pow(torch.minimum(cos, torch.tensor(1.)), 2))
        loss2 = sin.sum()
        mlflow.log_metrics({f"loss2_{i}": sin[i] for i in range(self.K)})
        # loss2 = -cos.sum()
        loss = loss2 + self.beta * loss1
        mlflow.log_metrics({"loss1":loss1.item(), "loss2":loss2.item()})
        return loss
    
    # def forward(self, data, label):
    #     output = self.student(data)
    #     base_loss = self.student.get_loss(output, label)
    #     kd_loss = self.get_loss(data, label)
    #     loss = self.lmbda * kd_loss
    #     return loss, base_loss.detach(), kd_loss.detach()
