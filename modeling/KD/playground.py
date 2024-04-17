import os
import math
import random
import numpy as np

from torch_cluster import knn
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Normalize, Expert, nosepExpert, pca
from .base_model import BaseKD


class CPD(BaseKD):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)

        self.num_experts = args.cpd_num_experts

        self.alpha = args.cpd_alpha
        self.beta = args.cpd_beta

        # For interesting item
        self.K = args.cpd_K
        self.T = args.cpd_T
        self.mxK = args.cpd_mxK
        self.tau_ce = args.cpd_tau_ce
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.get_topk_dict()
        ranking_list = torch.exp(-(torch.arange(self.mxK) + 1) / self.T)
        self.ranking_mat = ranking_list.repeat(self.num_users, 1)

        self.S_user_experts = nosepExpert(self.student_dim, self.teacher_dim, self.num_experts, norm=True)
        self.S_item_experts = nosepExpert(self.student_dim, self.teacher_dim, self.num_experts, norm=True)

    def get_topk_dict(self):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings()
            train_pairs = self.dataset.train_pairs
            # remove true interactions from topk_dict
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            _, self.topk_dict = torch.topk(inter_mat, self.mxK, dim=-1)
    
    def ce_ranking_loss(self, S, T):
        T_probs = torch.softmax(T / self.tau_ce, dim=-1)
        return F.cross_entropy(S / self.tau_ce, T_probs, reduction='sum')

    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def do_something_in_each_epoch(self, epoch):
        with torch.no_grad():
            # interesting items
            self.interesting_items = torch.zeros((self.num_users, self.K))

            # sampling
            while True:
                samples = torch.multinomial(self.ranking_mat, self.K, replacement=False)
                if (samples > self.mxK).sum() == 0:
                    break

            samples = samples.sort(dim=1)[0]

            for user in range(self.num_users):
                self.interesting_items[user] = self.topk_dict[user][samples[user]]

            self.interesting_items = self.interesting_items.cuda()

    def get_features(self, batch_entity, is_user, is_teacher, detach=False):
        model = self.teacher if is_teacher else self.student
        if is_user:
            x = model.get_user_embedding(batch_entity)
            experts = self.S_user_experts
        else:
            x = model.get_item_embedding(batch_entity)
            experts = self.S_item_experts
        
        if detach:
            x = x.detach()

        if is_teacher:
            feas = x     # batch_size x T_dim
        else:
            feas = experts(x)
        return feas

    def get_DE_loss(self, batch_entity, is_user):
        T_feas = self.get_features(batch_entity, is_user=is_user, is_teacher=True)
        S_feas = self.get_features(batch_entity, is_user=is_user, is_teacher=False)

        norm_T = T_feas.pow(2).sum(-1, keepdim=True).pow(1. / 2)
        T_feas = T_feas.div(norm_T)
        cos_theta = (T_feas * S_feas).sum(-1, keepdim=True)
        G_diff = 1. - cos_theta
        DE_loss = G_diff.sum()
        return DE_loss

    def get_reg(self, batch_user, batch_pos_item, batch_neg_item):
        # reg1
        post_u_feas = self.get_features(batch_user, is_user=True, is_teacher=False, detach=True).squeeze(1)
        post_pos_feas = self.get_features(batch_pos_item, is_user=False, is_teacher=False, detach=True).squeeze(1)
        post_neg_feas = self.get_features(batch_neg_item, is_user=False, is_teacher=False, detach=True).squeeze(1)
        post_pos_score = (post_u_feas * post_pos_feas).sum(-1)
        post_neg_score = (post_u_feas * post_neg_feas).sum(-1)
        pre_u_feas = self.student.get_user_embedding(batch_user).detach()
        pre_pos_feas = self.student.get_item_embedding(batch_pos_item).detach()
        pre_neg_feas = self.student.get_item_embedding(batch_neg_item).detach()
        pre_pos_score = (pre_u_feas * pre_pos_feas).sum(-1)
        pre_neg_score = (pre_u_feas * pre_neg_feas).sum(-1)
        reg1 = F.relu(-(post_pos_score - post_neg_score) * (pre_pos_score - pre_neg_score)).sum()
        
        # reg2
        topQ_items = self.interesting_items[batch_user].type(torch.LongTensor).cuda()
        batch_user_Q = batch_user.unsqueeze(-1)
        batch_user_Q = torch.cat(self.K * [batch_user_Q], 1)
        post_u = self.get_features(batch_user_Q, is_user=True, is_teacher=False, detach=True).squeeze()		# bs, Q, T_dim
        post_i = self.get_features(topQ_items, is_user=False, is_teacher=False, detach=True).squeeze()		# bs, Q, T_dim
        post_topQ_logits = (post_u * post_i).sum(dim=-1)         # bs, Q
        pre_topQ_logits = self.student.forward_multi_items(batch_user, topQ_items).detach()    # bs, Q
        reg2 = self.ce_ranking_loss(post_topQ_logits, pre_topQ_logits)

        return self.alpha * reg1 + self.beta * reg2

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        DE_loss_user = self.get_DE_loss(batch_user.unique(), is_user=True)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), is_user=False)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), is_user=False)
        DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5

        reg = self.get_reg(batch_user, batch_pos_item, batch_neg_item)

        loss = DE_loss + reg
        return loss


class NKD(BaseKD):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)

        self.num_experts = args.nkd_num_experts
        self.strategy = args.nkd_strategy
        self.alpha = args.nkd_alpha
        self.K = args.nkd_K
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.S_user_experts = nosepExpert(self.student_dim, self.teacher_dim, self.num_experts, norm=True)
        self.S_item_experts = nosepExpert(self.student_dim, self.teacher_dim, self.num_experts, norm=True)

        all_u, all_i = self.teacher.get_all_embedding()
        self.nearestK_u = self.get_nearest_K(all_u, self.K)
        self.nearestK_i = self.get_nearest_K(all_i, self.K)
    
    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def get_nearest_K(self, embs, K):
        with torch.no_grad():
            embs = pca(embs, 150)
            topk_indices = knn(embs, embs, k=K+1)[1].reshape(-1, K + 1)
        return topk_indices[:, 1:].cuda()

    def get_features(self, batch_entity, is_user):
        N = batch_entity.shape[0]
        if is_user:
            T = self.teacher.get_user_embedding(batch_entity)
            S = self.student.get_user_embedding(batch_entity)
            experts = self.S_user_experts
        else:
            T = self.teacher.get_item_embedding(batch_entity)
            S = self.student.get_item_embedding(batch_entity)
            experts = self.S_item_experts
        
        rnd_choice = torch.randint(0, self.K, (N, ), device='cuda')
        if is_user:
            neighborsS = self.student.get_user_embedding(self.nearestK_u[batch_entity, rnd_choice])     # bs, S_dim
            neighborsT = self.teacher.get_user_embedding(self.nearestK_u[batch_entity, rnd_choice])     # bs, T_dim
        else:
            neighborsS = self.student.get_item_embedding(self.nearestK_i[batch_entity, rnd_choice])     # bs, S_dim
            neighborsT = self.teacher.get_item_embedding(self.nearestK_i[batch_entity, rnd_choice])     # bs, T_dim

        if self.strategy == 'soft':
            rndS = torch.rand_like(S, device='cuda') * self.alpha * 2
            rndT = torch.rand_like(T, device='cuda') * self.alpha * 2
            S = rndS * S + (1. - rndS) * neighborsS     # bs, S_dim
            T = rndT * T + (1. - rndT) * neighborsT     # bs, T_dim
        elif self.strategy == 'hard':
            rndS = torch.rand_like(S, device='cuda')
            rndT = torch.rand_like(T, device='cuda')
            S = torch.where(rndS < self.alpha, S, neighborsS)   # bs, S_dim
            T = torch.where(rndT < self.alpha, T, neighborsT)   # bs, T_dim
        elif self.strategy == 'mix':
            S = self.alpha * S + (1. - self.alpha) * neighborsS
            T = self.alpha * T + (1. - self.alpha) * neighborsT
        elif self.strategy == 'randmix':
            rndS = random.random() * self.alpha * 2
            rndT = random.random() * self.alpha * 2
            S = rndS * S + (1. - rndS) * neighborsS
            T = rndT * T + (1. - rndT) * neighborsT
        elif self.strategy == 'hardmix':
            rndS = random.random()
            if rndS >= self.alpha:
                S = neighborsS
            rndT = random.random()
            if rndT >= self.alpha:
                T = neighborsT
        else:
            raise NotImplementedError
        
        S = experts(S)
        return T, S
    
    def get_DE_loss(self, batch_entity, is_user):
        T_feas, S_feas = self.get_features(batch_entity, is_user=is_user)

        norm_T = T_feas.pow(2).sum(-1, keepdim=True).pow(1. / 2)
        T_feas = T_feas.div(norm_T)
        cos_theta = (T_feas * S_feas).sum(-1, keepdim=True)
        G_diff = 1. - cos_theta
        DE_loss = G_diff.sum()
        return DE_loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        DE_loss_user = self.get_DE_loss(batch_user.unique(), is_user=True)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), is_user=False)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), is_user=False)

        DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5
        return DE_loss


class GraphD(BaseKD):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)

        self.num_experts = args.graphd_num_experts
        self.alpha = args.graphd_alpha
        self.K = args.graphd_K
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.S_user_experts = nosepExpert(self.student_dim, self.teacher_dim, self.num_experts, norm=True)
        self.S_item_experts = nosepExpert(self.student_dim, self.teacher_dim, self.num_experts, norm=True)

        all_u, all_i = self.teacher.get_all_embedding()
        self.nearestK_u = self.get_nearest_K(all_u, self.K) # |U|, K
        self.nearestK_i = self.get_nearest_K(all_i, self.K) # |I|, K
        self.Graph_u = self.construct_knn_graph(self.nearestK_u)
        self.Graph_i = self.construct_knn_graph(self.nearestK_i)
    
    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def get_nearest_K(self, embs, K):
        with torch.no_grad():
            embs = pca(embs, 150)
            topk_indices = knn(embs, embs, k=K+1)[1].reshape(-1, K + 1)
        return topk_indices[:, 1:].cuda()
    
    def construct_knn_graph(self, neighbor_id):
        N, K = neighbor_id.shape
        row = torch.arange(N).repeat(K, 1).T.reshape(-1).cuda()
        col = neighbor_id.reshape(-1)
        index = torch.stack([row, col])
        data = torch.ones(index.size(-1)).cuda() * (1. - self.alpha) / K
        self_loop_idx = torch.stack([torch.arange(N), torch.arange(N)]).cuda()
        self_loop_data = torch.ones(self_loop_idx.size(-1)).cuda() * self.alpha
        index = torch.cat([index, self_loop_idx], dim=1)
        data = torch.cat([data, self_loop_data])
        Graph = torch.sparse_coo_tensor(index, data,
                                            torch.Size([N, N]), dtype=torch.float)
        Graph = Graph.coalesce()
        return Graph.cuda()
    
    def computer(self):
        S_users_emb = self.student.user_emb.weight
        S_items_emb = self.student.item_emb.weight
        S_users_emb = torch.sparse.mm(self.Graph_u, S_users_emb)
        S_items_emb = torch.sparse.mm(self.Graph_i, S_items_emb)
        return S_users_emb, S_items_emb

    def get_features(self, batch_entity, is_user, S_users_emb, S_items_emb):
        N = batch_entity.shape[0]
        if is_user:
            T = self.teacher.get_user_embedding(batch_entity)
            S = S_users_emb[batch_entity]
            experts = self.S_user_experts
        else:
            T = self.teacher.get_item_embedding(batch_entity)
            S = S_items_emb[batch_entity]
            experts = self.S_item_experts
        S = experts(S)
        return T, S
    
    def get_DE_loss(self, batch_entity, is_user, S_users_emb, S_items_emb):
        T_feas, S_feas = self.get_features(batch_entity, is_user, S_users_emb, S_items_emb)

        norm_T = T_feas.pow(2).sum(-1, keepdim=True).pow(1. / 2)
        T_feas = T_feas.div(norm_T)
        cos_theta = (T_feas * S_feas).sum(-1, keepdim=True)
        G_diff = 1. - cos_theta
        DE_loss = G_diff.sum()
        return DE_loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        S_users_emb, S_items_emb = self.computer()
        DE_loss_user = self.get_DE_loss(batch_user.unique(), True, S_users_emb, S_items_emb)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), False, S_users_emb, S_items_emb)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), False, S_users_emb, S_items_emb)

        DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5
        return DE_loss
