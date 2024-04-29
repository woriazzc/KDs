import os
import math
import random
import numpy as np

from torch_cluster import knn
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Projector, pca, load_pkls, dump_pkls, self_loop_graph
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

        self.S_user_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=True)
        self.S_item_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=True)

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
        self.model_name = "nkd"

        self.num_experts = args.nkd_num_experts
        self.strategy = args.nkd_strategy
        self.alpha = args.nkd_alpha
        self.K = args.nkd_K
        self.dropout_rate = args.nkd_dropout_rate
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.S_user_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=True, dropout_rate=self.dropout_rate)
        self.S_item_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=True, dropout_rate=self.dropout_rate)

        all_u, all_i = self.teacher.get_all_embedding()
        self.nearestK_u, self.nearestK_i = self.get_nearest_K(all_u, all_i, self.K)

    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def _KNN(self, embs, K):
        with torch.no_grad():
            embs = pca(embs, 150)
            topk_indices = knn(embs, embs, k=K+1)[1].reshape(-1, K + 1)
        return topk_indices[:, 1:].cuda()

    def get_nearest_K(self, all_u, all_i, K):
        f_nearestK_u = os.path.join("modeling", "KD", "crafts", self.model_name, f"nearest_{K}_u.pkl")
        f_nearestK_i = os.path.join("modeling", "KD", "crafts", self.model_name, f"nearest_{K}_i.pkl")
        sucflg, nearestK_u, nearestK_i = load_pkls(f_nearestK_u, f_nearestK_i)
        if not sucflg:
            nearestK_u = self._KNN(all_u, K)
            nearestK_i = self._KNN(all_i, K)
            dump_pkls((nearestK_u, f_nearestK_u), (nearestK_i, f_nearestK_i))
        return nearestK_u, nearestK_i
    
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
        elif self.strategy == 'batchmix':
            rndS = torch.rand((N, 1), device='cuda')
            rndT = torch.rand((N, 1), device='cuda')
            S = rndS * S + (1. - rndS) * neighborsS
            T = rndT * T + (1. - rndT) * neighborsT
        elif self.strategy == 'layermix':
            rndS = torch.rand((1, S.size(1)), device='cuda')
            rndT = torch.rand((1, T.size(1)), device='cuda')
            S = rndS * S + (1. - rndS) * neighborsS
            T = rndT * T + (1. - rndT) * neighborsT
        elif self.strategy == 'allmix':
            rndS = torch.rand((1, S.size(1)), device='cuda') * torch.rand((N, 1), device='cuda')
            rndT = torch.rand((1, T.size(1)), device='cuda') * torch.rand((N, 1), device='cuda')
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
        self.model_name = "graphd"

        self.num_experts = args.graphd_num_experts
        self.alpha = args.graphd_alpha
        self.K = args.graphd_K
        self.keep_prob = args.graphd_keep_prob
        self.dropout_rate = args.graphd_dropout_rate
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.S_user_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=True, dropout_rate=self.dropout_rate)
        self.S_item_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=True, dropout_rate=self.dropout_rate)

        all_u, all_i = self.teacher.get_all_embedding()
        nearestK_u, nearestK_i = self.get_nearest_K(all_u, all_i, self.K)
        self.Graph_u = self.construct_knn_graph(nearestK_u)
        self.Graph_i = self.construct_knn_graph(nearestK_i)
    
    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def _KNN(self, embs, K):
        with torch.no_grad():
            embs = pca(embs, 150)
            topk_indices = knn(embs, embs, k=K+1)[1].reshape(-1, K + 1)
        return topk_indices[:, 1:].cuda()

    def get_nearest_K(self, all_u, all_i, K):
        f_nearestK_u = os.path.join("modeling", "KD", "crafts", self.model_name, f"nearest_{K}_u.pkl")
        f_nearestK_i = os.path.join("modeling", "KD", "crafts", self.model_name, f"nearest_{K}_i.pkl")
        sucflg, nearestK_u, nearestK_i = load_pkls(f_nearestK_u, f_nearestK_i)
        if not sucflg:
            nearestK_u = self._KNN(all_u, K)
            nearestK_i = self._KNN(all_i, K)
            dump_pkls((nearestK_u, f_nearestK_u), (nearestK_i, f_nearestK_i))
        return nearestK_u, nearestK_i
    
    def construct_knn_graph(self, neighbor_id):
        N, K = neighbor_id.shape
        row = torch.arange(N).repeat(K, 1).T.reshape(-1).cuda()
        col = neighbor_id.reshape(-1)
        index = torch.stack([row, col])
        data = torch.ones(index.size(-1)).cuda() / K
        Graph = torch.sparse_coo_tensor(index, data,
                                            torch.Size([N, N]), dtype=torch.float)
        Graph = Graph.coalesce()
        return Graph.cuda()

    def _dropout_graph(self, Graph):
        size = Graph.size()
        assert size[0] == size[1]
        index = Graph.indices().t()
        values = Graph.values()
        random_index = torch.rand(len(values)) + self.keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / self.keep_prob

        self_loop_idx = torch.stack([torch.arange(size[0]), torch.arange(size[0])], dim=1).cuda()
        self_loop_data = torch.ones(self_loop_idx.size(0)).cuda()

        rndS = random.random() * self.alpha * 2
        rndT = random.random() * self.alpha * 2
        valuesS = torch.cat([values * (1. - rndS), self_loop_data * rndS])
        valuesT = torch.cat([values * (1. - rndT), self_loop_data * rndT])

        index = torch.cat([index, self_loop_idx], dim=0)

        droped_GraphS = torch.sparse_coo_tensor(index.t(), valuesS, size, dtype=torch.float)
        droped_GraphT = torch.sparse_coo_tensor(index.t(), valuesT, size, dtype=torch.float)
        return droped_GraphS, droped_GraphT

    def get_features(self, batch_entity, is_user):
        if is_user:
            T = self.teacher.user_emb.weight
            S = self.student.user_emb.weight
            experts = self.S_user_experts
            Graph = self.Graph_u
        else:
            T = self.teacher.item_emb.weight
            S = self.student.item_emb.weight
            experts = self.S_item_experts
            Graph = self.Graph_i
        
        droped_GraphS, droped_GraphT = self._dropout_graph(Graph)
        
        S = torch.sparse.mm(droped_GraphS, S)
        T = torch.sparse.mm(droped_GraphT, T)
        T = T[batch_entity]
        S = S[batch_entity]
        S = experts(S)
        return T, S
    
    def get_DE_loss(self, batch_entity, is_user):
        T_feas, S_feas = self.get_features(batch_entity, is_user)

        norm_T = T_feas.pow(2).sum(-1, keepdim=True).pow(1. / 2)
        T_feas = T_feas.div(norm_T)
        cos_theta = (T_feas * S_feas).sum(-1, keepdim=True)
        G_diff = 1. - cos_theta
        DE_loss = G_diff.sum()
        return DE_loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        DE_loss_user = self.get_DE_loss(batch_user.unique(), True)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), False)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), False)

        DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5
        return DE_loss


class FilterD(BaseKD):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "filterd"

        self.num_experts = args.filterd_num_experts
        self.alpha = args.filterd_alpha
        self.beta = args.filterd_beta
        self.K = args.filterd_K
        self.keep_prob = args.filterd_keep_prob
        self.dropout_rate = args.filterd_dropout_rate
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.S_user_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=True, dropout_rate=self.dropout_rate)
        self.S_item_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=True, dropout_rate=self.dropout_rate)

        all_u, all_i = self.teacher.get_all_embedding()
        nearestK_u, nearestK_i = self.get_nearest_K(all_u, all_i, self.K)
        self.Graph_u = self.construct_knn_graph(nearestK_u)
        self.Graph_i = self.construct_knn_graph(nearestK_i)
    
    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def _KNN(self, embs, K):
        with torch.no_grad():
            embs = pca(embs, 150)
            topk_indices = knn(embs, embs, k=K+1)[1].reshape(-1, K + 1)
        return topk_indices[:, 1:].cuda()

    def get_nearest_K(self, all_u, all_i, K):
        f_nearestK_u = os.path.join("modeling", "KD", "crafts", self.model_name, f"nearest_{K}_u.pkl")
        f_nearestK_i = os.path.join("modeling", "KD", "crafts", self.model_name, f"nearest_{K}_i.pkl")
        sucflg, nearestK_u, nearestK_i = load_pkls(f_nearestK_u, f_nearestK_i)
        if not sucflg:
            nearestK_u = self._KNN(all_u, K)
            nearestK_i = self._KNN(all_i, K)
            dump_pkls((nearestK_u, f_nearestK_u), (nearestK_i, f_nearestK_i))
        return nearestK_u, nearestK_i
    
    def construct_knn_graph(self, neighbor_id):
        N, K = neighbor_id.shape
        row = torch.arange(N).repeat(K, 1).T.reshape(-1).cuda()
        col = neighbor_id.reshape(-1)
        index = torch.stack([row, col])
        data = torch.ones(index.size(-1)).cuda() / K
        Graph = torch.sparse_coo_tensor(index, data,
                                            torch.Size([N, N]), dtype=torch.float)
        Graph = Graph.coalesce()
        return Graph.cuda()
    
    def _sym_normalize(self, Graph):
        dense = Graph.to_dense().cpu()
        D = torch.sum(dense, dim=1).float()
        D[D == 0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense / D_sqrt.t()
        index = dense.nonzero(as_tuple=False)
        data = dense[dense != 0]
        assert len(index) == len(data)
        Graph = torch.sparse_coo_tensor(index.t(), data, torch.Size(Graph.size()), dtype=torch.float)
        return Graph.coalesce().cuda()

    def _dropout_graph(self, Graph):
        size = Graph.size()
        assert size[0] == size[1]
        index = Graph.indices().t()
        values = Graph.values()
        random_index = torch.rand(len(values)) + self.keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / self.keep_prob

        self_loop_idx = torch.stack([torch.arange(size[0]), torch.arange(size[0])], dim=1).cuda()
        self_loop_data = torch.ones(self_loop_idx.size(0)).cuda()

        rndS = random.random() * self.alpha * 2
        rndT = random.random() * self.alpha * 2
        valuesS = torch.cat([values * (1. - rndS), self_loop_data * rndS])
        valuesT = torch.cat([values * (1. - rndT), self_loop_data * rndT])

        index = torch.cat([index, self_loop_idx], dim=0)

        droped_GraphS = torch.sparse_coo_tensor(index.t(), valuesS, size, dtype=torch.float)
        droped_GraphT = torch.sparse_coo_tensor(index.t(), valuesT, size, dtype=torch.float)

        # droped_GraphS = self._sym_normalize(droped_GraphS)
        # droped_GraphT = self._sym_normalize(droped_GraphT)

        return droped_GraphS, droped_GraphT

    def get_features(self, batch_entity, is_user):
        if is_user:
            T = self.teacher.user_emb.weight
            S = self.student.user_emb.weight
            experts = self.S_user_experts
            Graph = self.Graph_u
        else:
            T = self.teacher.item_emb.weight
            S = self.student.item_emb.weight
            experts = self.S_item_experts
            Graph = self.Graph_i
        
        smooth_droped_GraphS, smooth_droped_GraphT = self._dropout_graph(Graph)
        rough_droped_GraphS = self_loop_graph(smooth_droped_GraphS.size(0)).cuda() - smooth_droped_GraphS
        rough_droped_GraphT = self_loop_graph(smooth_droped_GraphT.size(0)).cuda() - smooth_droped_GraphT
        
        smooth_S = torch.sparse.mm(smooth_droped_GraphS, S)
        smooth_T = torch.sparse.mm(smooth_droped_GraphT, T)
        smooth_T = smooth_T[batch_entity]
        smooth_S = smooth_S[batch_entity]
        smooth_S = experts(smooth_S)

        rough_S = torch.sparse.mm(rough_droped_GraphS, S)
        rough_T = torch.sparse.mm(rough_droped_GraphT, T)
        rough_T = rough_T[batch_entity]
        rough_S = rough_S[batch_entity]
        rough_S = experts(rough_S)
        return smooth_S, rough_S, smooth_T, rough_T
    
    def cal_FD_loss(self, T_feas, S_feas):
        norm_T = T_feas.pow(2).sum(-1, keepdim=True).pow(1. / 2)
        T_feas = T_feas.div(norm_T)
        cos_theta = (T_feas * S_feas).sum(-1, keepdim=True)
        G_diff = 1. - cos_theta
        FD_loss = G_diff.sum()
        return FD_loss
    
    def get_DE_loss(self, batch_entity, is_user):
        smooth_S, rough_S, smooth_T, rough_T = self.get_features(batch_entity, is_user)
        smooth_loss = self.cal_FD_loss(smooth_T, smooth_S)
        rough_loss = self.cal_FD_loss(rough_T, rough_S)
        DE_loss = smooth_loss + self.beta * rough_loss
        return DE_loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        DE_loss_user = self.get_DE_loss(batch_user.unique(), True)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), False)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), False)

        DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5
        return DE_loss


class FD(BaseKD):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "fd"

        self.num_experts = args.fd_num_experts
        self.alpha = args.fd_alpha
        self.beta = args.fd_beta
        self.K = args.fd_K
        self.smooth_ratio = args.fd_smooth_ratio
        self.rough_ratio = args.fd_rough_ratio
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.S_user_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=True)
        self.S_item_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=True)

        self.Graph_smooth, self.Graph_rough = self.construct_graph(self.smooth_ratio, self.rough_ratio)
        self.T_users_emb_smooth, self.T_items_emb_smooth = self.computer(self.teacher.user_emb.weight, self.teacher.item_emb.weight, self.Graph_smooth)
        self.T_users_emb_rough, self.T_items_emb_rough = self.computer(self.teacher.user_emb.weight, self.teacher.item_emb.weight, self.Graph_rough)
    
    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def construct_graph(self, smooth_ratio, rough_ratio):
        user_dim = torch.LongTensor(self.teacher.dataset.train_pairs[:, 0].cpu())
        item_dim = torch.LongTensor(self.teacher.dataset.train_pairs[:, 1].cpu())

        first_sub = torch.stack([user_dim, item_dim + self.num_users])
        second_sub = torch.stack([item_dim + self.num_users, user_dim])
        index = torch.cat([first_sub, second_sub], dim=1)
        data = torch.ones(index.size(-1)).int()
        Graph = torch.sparse_coo_tensor(index, data,
                                            torch.Size([self.num_users + self.num_items, self.num_users + self.num_items]), dtype=torch.int)
        dense = Graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D == 0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense / D_sqrt.t()
        index = dense.nonzero(as_tuple=False)
        data = dense[dense >= 1e-9]
        assert len(index) == len(data)
        Graph = torch.sparse_coo_tensor(index.t(), data, torch.Size(
            [self.num_users + self.num_items, self.num_users + self.num_items]), dtype=torch.float)
        Graph = Graph.coalesce()

        smooth_dim = int(Graph.shape[0] * smooth_ratio)
        f_smooth_values = os.path.join("modeling", "KD", "crafts", "fd", f"smooth_values_{smooth_dim}.pkl")
        f_smooth_vectors = os.path.join("modeling", "KD", "crafts", "fd", f"smooth_vectors_{smooth_dim}.pkl")
        sucflg, smooth_values, smooth_vectors = load_pkls(f_smooth_values, f_smooth_vectors)
        if not sucflg:
            if smooth_ratio <= 0.3:
                smooth_values, smooth_vectors = torch.lobpcg(Graph, k=smooth_dim, largest=True, niter=5)
            else:
                smooth_vectors, smooth_values, _ = torch.svd_lowrank(Graph, q=smooth_dim, niter=10)
            dump_pkls((smooth_values, f_smooth_values), (smooth_vectors, f_smooth_vectors))
        rough_dim = int(Graph.shape[0] * rough_ratio)
        rough_values, rough_vectors = torch.lobpcg(Graph, k=rough_dim, largest=False, niter=5)
        smooth_filter = (smooth_vectors * self.weight_feature(smooth_values)).mm(smooth_vectors.t())
        rough_filter = (rough_vectors * self.weight_feature(rough_values)).mm(rough_vectors.t())
        return smooth_filter.cuda(), rough_filter.cuda()
    
    def weight_feature(self, value):
        return torch.exp(self.beta * value)
        # return value
    
    def computer(self, users_emb, items_emb, Graph):
        all_embs = torch.cat([users_emb, items_emb])
        filtered_embs = torch.mm(Graph, all_embs)
        filtered_users_emb, filtered_items_emb = torch.split(filtered_embs, [self.num_users, self.num_items])
        return filtered_users_emb, filtered_items_emb

    def get_features(self, batch_entity, is_user, S_emb, T_emb):
        N = batch_entity.shape[0]
        T = T_emb[batch_entity]
        S = S_emb[batch_entity]
        if is_user:
            experts = self.S_user_experts
        else:
            experts = self.S_item_experts
        S = experts(S)
        return T, S
    
    def get_DE_loss(self, batch_entity, is_user, S_emb, T_emb):
        T_feas, S_feas = self.get_features(batch_entity, is_user, S_emb, T_emb)

        norm_T = T_feas.pow(2).sum(-1, keepdim=True).pow(1. / 2)
        T_feas = T_feas.div(norm_T)
        cos_theta = (T_feas * S_feas).sum(-1, keepdim=True)
        G_diff = 1. - cos_theta
        DE_loss = G_diff.sum()
        return DE_loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        S_users_emb_smooth, S_items_emb_smooth = self.computer(self.student.user_emb.weight, self.student.item_emb.weight, self.Graph_smooth)
        S_users_emb_rough, S_items_emb_rough = self.computer(self.student.user_emb.weight, self.student.item_emb.weight, self.Graph_rough)

        DE_loss_user = self.get_DE_loss(batch_user.unique(), True, S_users_emb_smooth, self.T_users_emb_smooth)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), False, S_items_emb_smooth, self.T_items_emb_smooth)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), False, S_items_emb_smooth, self.T_items_emb_smooth)
        DE_loss_smooth = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5

        DE_loss_user = self.get_DE_loss(batch_user.unique(), True, S_users_emb_rough, self.T_users_emb_rough)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), False, S_items_emb_rough, self.T_items_emb_rough)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), False, S_items_emb_rough, self.T_items_emb_rough)
        DE_loss_rough = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5

        DE_loss = self.alpha * DE_loss_smooth + (1. - self.alpha) * DE_loss_rough
        return DE_loss


class GDCP(GraphD):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)

        self.reg_type = args.gdcp_reg_type
        self.dec_epoch = args.gdcp_dec_epoch
        self.init_w_reg = args.gdcp_w_reg
        self.w_reg = self.init_w_reg

        if self.reg_type == "first":
            self.margin = args.gdcp_margin
        elif self.reg_type == "second":
            self.Q = args.gdcp_Q
            self.T = args.gdcp_T
            self.mxK = args.gdcp_mxK
            self.tau_ce = args.gdcp_tau_ce
            self.topk_dict = self.get_topk_dict()
            ranking_list = torch.exp(-(torch.arange(self.mxK) + 1) / self.T)
            self.ranking_mat = ranking_list.repeat(self.num_users, 1)
        else:
            raise NotImplementedError

    def get_topk_dict(self):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings()
            train_pairs = self.dataset.train_pairs
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            _, topk_dict = torch.topk(inter_mat, self.mxK, dim=-1)
        return topk_dict
    
    def ce_ranking_loss(self, S, T):
        T_probs = torch.softmax(T / self.tau_ce, dim=-1)
        return F.cross_entropy(S / self.tau_ce, T_probs, reduction='sum')

    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def do_something_in_each_epoch(self, epoch):
        self.w_reg = max(0., 1. - epoch / self.dec_epoch) * self.init_w_reg
        
        if self.reg_type == "second":
            with torch.no_grad():
                self.interesting_items = torch.zeros((self.num_users, self.Q))
                while True:
                    samples = torch.multinomial(self.ranking_mat, self.Q, replacement=False)
                    if (samples > self.mxK).sum() == 0:
                        break
                samples = samples.sort(dim=1)[0]
                for user in range(self.num_users):
                    self.interesting_items[user] = self.topk_dict[user][samples[user]]
                self.interesting_items = self.interesting_items.cuda()

    def get_features(self, batch_entity, is_user, is_reg=False):
        if is_user:
            T = self.teacher.user_emb.weight
            S = self.student.user_emb.weight
            experts = self.S_user_experts
            Graph = self.Graph_u
        else:
            T = self.teacher.item_emb.weight
            S = self.student.item_emb.weight
            experts = self.S_item_experts
            Graph = self.Graph_i
        
        droped_GraphS, droped_GraphT = self._dropout_graph(Graph)
        
        S = torch.sparse.mm(droped_GraphS, S)
        T = torch.sparse.mm(droped_GraphT, T)
        T = T[batch_entity]
        S = S[batch_entity]
        # if is_reg:
        #     pre_S = S.detach().clone()
        # else:
        #     pre_S = S
        pre_S = S
        post_S = experts(pre_S)
        if is_reg:
            return pre_S, post_S
        else:
            return T, post_S
    
    def get_reg(self, batch_user, batch_pos_item, batch_neg_item):
        if self.reg_type == "first":
            post_u_feas, pre_u_feas = self.get_features(batch_user, is_user=True, is_reg=True)
            post_pos_feas, pre_pos_feas = self.get_features(batch_pos_item, is_user=False, is_reg=True)
            post_neg_feas, pre_neg_feas = self.get_features(batch_neg_item, is_user=False, is_reg=True)

            post_pos_score = (post_u_feas * post_pos_feas).sum(-1, keepdim=True)
            post_neg_score = torch.bmm(post_neg_feas, post_u_feas.unsqueeze(-1)).squeeze(-1)
            post_pos_score = post_pos_score.expand_as(post_neg_score)

            pre_pos_score = (pre_u_feas * pre_pos_feas).sum(-1, keepdim=True)
            pre_neg_score = torch.bmm(pre_neg_feas, pre_u_feas.unsqueeze(-1)).squeeze(-1)
            pre_pos_score = pre_pos_score.expand_as(pre_neg_score)
            reg = F.relu(-(post_pos_score - post_neg_score) * (pre_pos_score - pre_neg_score) - self.margin).mean(1).sum()
        else:
            topQ_items = self.interesting_items[batch_user].type(torch.LongTensor).cuda()
            post_u_feas, pre_u_feas = self.get_features(batch_user, is_user=True, is_reg=True)		# bs, S_dim
            post_i_feas, pre_i_feas = self.get_features(topQ_items, is_user=False, is_reg=True)		# bs, Q, S_dim
            post_topQ_logits = torch.bmm(post_i_feas, post_u_feas.unsqueeze(-1)).squeeze(-1)    # bs, Q
            pre_topQ_logits = torch.bmm(pre_i_feas, pre_u_feas.unsqueeze(-1)).squeeze(-1)    # bs, Q
            reg = self.ce_ranking_loss(post_topQ_logits, pre_topQ_logits)
        return reg

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        DE_loss_user = self.get_DE_loss(batch_user.unique(), True)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), False)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), False)
        DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5

        if self.w_reg > 0:
            reg = self.get_reg(batch_user, batch_pos_item, batch_neg_item)
        else:
            reg = 0.
        
        loss = DE_loss + reg * self.w_reg
        return loss
