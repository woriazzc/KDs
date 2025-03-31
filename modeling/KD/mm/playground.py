import os
import math
import mlflow
import random
import numpy as np
from copy import deepcopy

from torch_cluster import knn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils import Projector, graph_mm, pca, load_pkls, dump_pkls, self_loop_graph, sym_norm_graph, MetaOptimizer
from ..base_model import BaseKD4MM


class FreqMM(BaseKD4MM):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "freqmm"
        self.norm = args.norm
        self.alpha = args.freqmm_alpha
        self.K = args.freqmm_K
        self.keep_prob = args.freqmm_keep_prob
        self.dropout_rate = args.freqmm_dropout_rate
        self.hidden_dim_ratio = args.hidden_dim_ratio
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.all_T_u, self.all_T_i = self.teacher.get_all_embedding()
        
        self.projector_u = Projector(self.student_dim, self.teacher_dim, 1, norm=False, dropout_rate=self.dropout_rate, hidden_dim_ratio=self.hidden_dim_ratio)
        self.projector_i = Projector(self.student_dim, self.teacher_dim, 1, norm=False, dropout_rate=self.dropout_rate, hidden_dim_ratio=self.hidden_dim_ratio)
        self.Graph_u = self.construct_knn_graph(self.all_T_u, self.K, "user")
        self.Graph_i = self.construct_knn_graph(self.all_T_i, self.K, "item")
    
    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    @torch.no_grad()
    def _KNN(self, embs, K):
        embs = pca(embs, 150)
        topk_indices = knn(embs, embs, k=K+1)[1].reshape(-1, K + 1)
        return topk_indices[:, 1:].cuda()
    
    def construct_knn_graph(self, all_embed, K, name):
        f_graph = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.T_backbone, self.model_name, f"Graph_{name}_{K}.pkl")
        sucflg, Graph = load_pkls(f_graph)
        if sucflg: return Graph.cuda()
        N = all_embed.shape[0]
        nearestK = self._KNN(all_embed, K)
        row = torch.arange(N).repeat(K, 1).T.reshape(-1).cuda()
        col = nearestK.reshape(-1)
        index = torch.stack([row, col])
        data = torch.ones(index.size(-1)).cuda()
        Graph = torch.sparse_coo_tensor(index, data, torch.Size([N, N]), dtype=torch.float)
        Graph = (Graph + Graph.T) / 2
        Graph = Graph.coalesce()
        Graph = sym_norm_graph(Graph)
        dump_pkls((Graph, f_graph))
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
        droped_Graph = torch.sparse_coo_tensor(index.t(), values, size, dtype=torch.float)
        return droped_Graph

    def generate_filter(self, Adj, alpha):
        droped_Adj = self._dropout_graph(Adj)
        self_loop = self_loop_graph(droped_Adj.shape[0]).cuda()
        H = droped_Adj * alpha + self_loop * (1. - alpha)
        return H
    
    def freq_loss(self, batch_entity, all_S, all_T, filter, projector):
        proj_all_S = projector(all_S)
        S = torch.sparse.mm(filter, proj_all_S)
        T = torch.sparse.mm(filter, all_T)
        T_feas = T[batch_entity]
        S_feas = S[batch_entity]
        if self.norm:
            T_feas = F.normalize(T_feas, p=2, dim=-1)
            S_feas = F.normalize(S_feas, p=2, dim=-1)
            cos_theta = (T_feas * S_feas).sum(-1, keepdim=True)
            G_diff = 1. - cos_theta
        else:
            G_diff = (T_feas - S_feas).pow(2).sum(-1)
        loss = G_diff.sum()
        return loss

    def get_kd_loss(self, batch_entity, mode):
        if mode == "user":
            all_T = self.all_T_u
            all_S = self.student.user_emb.weight
            projector = self.projector_u
            H = self.generate_filter(self.Graph_u, self.alpha)
        else:
            all_T = self.all_T_i
            all_S = self.student.item_emb.weight
            H = self.generate_filter(self.Graph_i, self.alpha)
            projector = self.projector_i
        loss = self.freq_loss(batch_entity, all_S, all_T, H, projector)
        return loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        batch_item = torch.cat([batch_pos_item.unique(), batch_neg_item.unique()]).unique()
        loss_user = self.get_kd_loss(batch_user.unique(), mode="user")
        loss_item = self.get_kd_loss(batch_item, mode="item")
        loss = loss_user + loss_item
        return loss

    def do_something_in_each_epoch(self, epoch):
        with torch.no_grad():
            suff = "_norm" if self.args.norm else ""
            torch.save(self.all_T_u.detach().cpu(), f"T_u{suff}.pt")
            torch.save(self.all_T_i.detach().cpu(), f"T_i{suff}.pt")
            S_u = self.projector_u(self.student.user_emb.weight)
            S_i = self.projector_i(self.student.item_emb.weight)
            torch.save(S_u.detach().cpu(), f"S_u{suff}.pt")
            torch.save(S_i.detach().cpu(), f"S_i{suff}.pt")


class IdealD(BaseKD4MM):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "ideald"
        self.filter_id = args.filter_id
        self.norm = args.norm
        self.K = args.ideald_K
        self.dropout_rate = args.ideald_dropout_rate
        self.hidden_dim_ratio = args.hidden_dim_ratio
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim
        self.projector_u = Projector(self.student_dim, self.teacher_dim, 1, norm=False, dropout_rate=self.dropout_rate, hidden_dim_ratio=self.hidden_dim_ratio)
        self.projector_i = Projector(self.student_dim, self.teacher_dim, 1, norm=False, dropout_rate=self.dropout_rate, hidden_dim_ratio=self.hidden_dim_ratio)
        self.all_T_u, self.all_T_i = self.teacher.get_all_embedding()
        Graph_u = self.construct_knn_graph(self.all_T_u, self.K, "user")
        Graph_i = self.construct_knn_graph(self.all_T_i, self.K, "item")
        U_user = self.eig_decompose(Graph_u, "user")
        U_item = self.eig_decompose(Graph_i, "item")
        self.anchor_filters_user = self.construct_anchor_filters(U_user)
        self.anchor_filters_item = self.construct_anchor_filters(U_item)
        if self.filter_id == -1:
            self.filter_u = self_loop_graph(self.num_users).cuda()
            self.filter_i = self_loop_graph(self.num_items).cuda()
        else:
            self.filter_u = self.anchor_filters_user[self.filter_id]
            self.filter_i = self.anchor_filters_item[self.filter_id]
        self.global_step = 0

    @torch.no_grad()
    def _KNN(self, embs, K):
        embs = pca(embs, 150)
        topk_indices = knn(embs, embs, k=K+1)[1].reshape(-1, K + 1)
        return topk_indices[:, 1:].cuda()
    
    def construct_knn_graph(self, all_embed, K, name):
        f_graph = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.T_backbone, self.model_name, f"Graph_{name}_{K}.pkl")
        sucflg, Graph = load_pkls(f_graph)
        if sucflg: return Graph.cuda()
        N = all_embed.shape[0]
        nearestK = self._KNN(all_embed, K)
        row = torch.arange(N).repeat(K, 1).T.reshape(-1).cuda()
        col = nearestK.reshape(-1)
        index = torch.stack([row, col])
        data = torch.ones(index.size(-1)).cuda()
        Graph = torch.sparse_coo_tensor(index, data, torch.Size([N, N]), dtype=torch.float)
        Graph = (Graph + Graph.T) / 2
        Graph = Graph.coalesce()
        Graph = sym_norm_graph(Graph)
        dump_pkls((Graph, f_graph))
        return Graph.cuda()
    
    def eig_decompose(self, Graph, name):
        f_U = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.T_backbone, self.model_name, f"U_{name}_{self.K}.pkl")
        sucflg, U = load_pkls(f_U)
        if sucflg: return U.cuda()
        U, S, V = torch.svd(Graph.to_dense())
        U = U.real
        dump_pkls((U, f_U))
        return U.cuda()

    def construct_anchor_filters(self, U):
        anchor_filters = []
        blk = math.ceil(U.shape[1] / 4)
        for i in range(4):
            vectors = U[:, blk * i:blk * (i + 1)]
            filter = vectors.mm(vectors.t())
            anchor_filters.append(filter.cuda())
        return anchor_filters

    def get_features(self, batch_entity, is_user):
        if is_user:
            T = self.all_T_u
            S = self.student.user_emb.weight
            proj = self.projector_u
            Graph = self.filter_u
        else:
            T = self.all_T_i
            S = self.student.item_emb.weight
            proj = self.projector_u
            Graph = self.filter_i
        SP = proj(S)
        filtered_S = torch.sparse.mm(Graph, SP)
        filtered_T = torch.sparse.mm(Graph, T)
        filtered_T = filtered_T[batch_entity]
        filtered_S = filtered_S[batch_entity]
        if self.norm:
            filtered_S = F.normalize(filtered_S, p=2, dim=-1)
            filtered_T = F.normalize(filtered_T, p=2, dim=-1)
        return filtered_T, filtered_S
    
    def get_DE_loss(self, batch_entity, is_user):
        T_feas, S_feas = self.get_features(batch_entity, is_user)
        if self.norm:
            G_diff = 1. - (T_feas * S_feas).sum(-1, keepdim=True)
        else:
            G_diff = (T_feas - S_feas).pow(2).sum(-1)
        loss = G_diff.sum()
        return loss
    
    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        batch_item = torch.cat([batch_pos_item.unique(), batch_neg_item.unique()]).unique()
        loss_user = self.get_DE_loss(batch_user, True)
        loss_item = self.get_DE_loss(batch_item, False)
        loss = loss_user + loss_item
        self.log_anchor_loss(batch_user, True)
        return loss

    @torch.no_grad()
    def log_anchor_loss(self, batch_entity, is_user):
        if is_user:
            T = self.all_T_u
            S = self.student.user_emb.weight
            proj = self.projector_u
            anchor_filters = self.anchor_filters_user
        else:
            T = self.all_T_i
            S = self.student.item_emb.weight
            proj = self.projector_i
            anchor_filters = self.anchor_filters_item
        for idx, filter in enumerate(anchor_filters):
            filtered_S = torch.sparse.mm(filter, S)
            filtered_T = torch.sparse.mm(filter, T)
            filtered_T = filtered_T[batch_entity]
            filtered_S = filtered_S[batch_entity]
            training = proj.training
            proj.eval()
            filtered_SP = proj(filtered_S)
            proj.train(training)
            if self.norm:
                filtered_SP = F.normalize(filtered_SP, p=2, dim=-1)
                filtered_T = F.normalize(filtered_T, p=2, dim=-1)
                G_diff = 1. - (filtered_T * filtered_SP).sum(-1, keepdim=True)
            else:
                G_diff = (filtered_T - filtered_SP).pow(2).sum(-1) / 2
            loss = G_diff.sum()
            mlflow.log_metrics({f"anchor_loss_{idx}": (loss / batch_entity.shape[0]).cpu().item()}, step=self.global_step)
        self.global_step += 1


class AdaFreq(BaseKD4MM):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "adafreq"
        self.norm = args.norm
        self.K = args.adafreq_K
        self.keep_prob = args.adafreq_keep_prob
        self.dropout_rate = args.adafreq_dropout_rate
        self.hidden_dim_ratio = args.hidden_dim_ratio
        self.beta = args.adafreq_beta
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.all_T_u, self.all_T_i = self.teacher.get_all_embedding()
        
        self.projector_u = Projector(self.student_dim, self.teacher_dim, 1, norm=False, dropout_rate=self.dropout_rate, hidden_dim_ratio=self.hidden_dim_ratio)
        self.projector_i = Projector(self.student_dim, self.teacher_dim, 1, norm=False, dropout_rate=self.dropout_rate, hidden_dim_ratio=self.hidden_dim_ratio)
        self.Graph_u = self.construct_knn_graph(self.all_T_u, self.K, "user")
        self.Graph_i = self.construct_knn_graph(self.all_T_i, self.K, "item")
        self.gate_u = nn.Sequential(
            nn.Linear(self.teacher_dim, self.teacher_dim // 2),
            nn.ReLU(),
            nn.Linear(self.teacher_dim // 2, 1),
        )
        self.gate_i = nn.Sequential(
            nn.Linear(self.teacher_dim, self.teacher_dim // 2),
            nn.ReLU(),
            nn.Linear(self.teacher_dim // 2, 1),
        )
    
    @torch.no_grad()
    def _KNN(self, embs, K):
        embs = pca(embs, 150)
        topk_indices = knn(embs, embs, k=K+1)[1].reshape(-1, K + 1)
        return topk_indices[:, 1:].cuda()
    
    def construct_knn_graph(self, all_embed, K, name):
        f_graph = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.T_backbone, self.model_name, f"Graph_{name}_{K}.pkl")
        sucflg, Graph = load_pkls(f_graph)
        if sucflg: return Graph.cuda()
        N = all_embed.shape[0]
        nearestK = self._KNN(all_embed, K)
        row = torch.arange(N).repeat(K, 1).T.reshape(-1).cuda()
        col = nearestK.reshape(-1)
        index = torch.stack([row, col])
        data = torch.ones(index.size(-1)).cuda()
        Graph = torch.sparse_coo_tensor(index, data, torch.Size([N, N]), dtype=torch.float)
        Graph = (Graph + Graph.T) / 2
        Graph = Graph.coalesce()
        Graph = sym_norm_graph(Graph)
        dump_pkls((Graph, f_graph))
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
        droped_Graph = torch.sparse_coo_tensor(index.t(), values, size, dtype=torch.float)
        return droped_Graph

    def generate_filter(self, Adj, mode):
        droped_Adj = self._dropout_graph(Adj)
        if mode == "user":
            alpha = self.gate_u(self.all_T_u).squeeze()
        else:
            alpha = self.gate_i(self.all_T_i).squeeze()
        alpha = torch.sigmoid(alpha) * 0.5
        torch.save(alpha.detach().cpu(), f"alpha_{mode}.pt")
        self_loop_alpha = self_loop_graph(droped_Adj.shape[0], alpha).cuda()
        self_loop_one = self_loop_graph(droped_Adj.shape[0]).cuda()
        H = graph_mm(self_loop_alpha, droped_Adj) + self_loop_one - self_loop_alpha
        return H
    
    def freq_loss(self, batch_entity, all_S, all_T, filter, projector):
        proj_all_S = projector(all_S)
        S = graph_mm(filter, proj_all_S)
        T = graph_mm(filter, all_T)
        T_feas = T[batch_entity]
        S_feas = S[batch_entity]
        if self.norm:
            T_feas = F.normalize(T_feas, p=2, dim=-1)
            S_feas = F.normalize(S_feas, p=2, dim=-1)
            cos_theta = (T_feas * S_feas).sum(-1, keepdim=True)
            G_diff = 1. - cos_theta
        else:
            G_diff = (T_feas - S_feas).pow(2).sum(-1)
        loss = G_diff.sum()
        return loss

    def get_user_loss(self, batch_entity):
        all_T = self.all_T_u
        all_S = self.student.user_emb.weight
        H_u = self.generate_filter(self.Graph_u, "user")
        loss = self.freq_loss(batch_entity, all_S, all_T, H_u, self.projector_u)
        return loss
    
    def get_item_loss(self, batch_entity):
        all_T = self.all_T_i
        all_S_id = self.student.item_emb.weight
        H_i = self.generate_filter(self.Graph_i, "item")
        loss = self.freq_loss(batch_entity, all_S_id, all_T, H_i, self.projector_i)
        return loss

    def get_filter_loss(self, batch_user, batch_pos_item, batch_neg_item, mode):
        H_u = self.generate_filter(self.Graph_u, "user")
        H_i = self.generate_filter(self.Graph_i, "item")
        if mode == "T":
            all_u_ori, all_i_ori = self.all_T_u, self.all_T_i
        else:
            all_u_ori = self.student.user_emb.weight.detach()
            all_i_ori = self.student.item_emb.weight.detach()
        all_u = graph_mm(H_u, all_u_ori)
        u = all_u[batch_user]
        all_i = graph_mm(H_i, all_i_ori)
        i = all_i[batch_pos_item]
        j = all_i[batch_neg_item]
        pos_score = (u * i).sum(dim=1, keepdim=True)    # batch_size, 1
        neg_score = torch.bmm(j, u.unsqueeze(-1)).squeeze(-1)   # batch_size, num_ns
        pos_score = pos_score.expand_as(neg_score)  # batch_size, num_ns
        loss = -F.logsigmoid(pos_score - neg_score).mean(1).sum()
        return loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        batch_item = torch.cat([batch_pos_item.unique(), batch_neg_item.unique()]).unique()
        loss_user = self.get_user_loss(batch_user.unique())
        loss_item = self.get_item_loss(batch_item)
        loss_filter_T = self.get_filter_loss(batch_user, batch_pos_item, batch_neg_item, mode="T")
        loss_filter_S = self.get_filter_loss(batch_user, batch_pos_item, batch_neg_item, mode="S")
        loss = loss_user + loss_item + self.beta * (loss_filter_T + loss_filter_S)
        return loss


class nodeMM(BaseKD4MM):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "nodemm"
    
        self.K = args.nodemm_K
        self.dropout_rate = args.nodemm_dropout_rate
        self.L = args.nodemm_L  # order of Chebyshev Polynomial
        self.d = args.nodemm_d  # size of codebook
        self.alpha = args.nodemm_alpha  # for PPR initialization
        self.f = args.nodemm_f  # transformed dimension
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.all_T_u, self.all_T_i = self.teacher.get_all_embedding()
        
        Graph_u = self.construct_knn_graph(self.all_T_u, self.K, "user")
        Graph_i = self.construct_knn_graph(self.all_T_i, self.K, "item")
        self.L_u = (self_loop_graph(self.num_users).cuda() - Graph_u).coalesce()
        self.L_i = (self_loop_graph(self.num_items).cuda() - Graph_i).coalesce()
        self.gamma_u = torch.nn.Parameter(torch.zeros((self.d, self.L + 1)))
        self.gamma_i = torch.nn.Parameter(torch.zeros((self.d, self.L + 1)))
        for k in range(self.L + 1):
            self.gamma_u.data[:, k] = self.alpha * (1 - self.alpha) ** k
            self.gamma_i.data[:, k] = self.alpha * (1 - self.alpha) ** k
        self.gamma_u.data[:, -1] = (1 - self.alpha) ** self.L
        self.gamma_i.data[:, -1] = (1 - self.alpha) ** self.L
        self.projs_u = nn.ModuleList([nn.Linear(self.f, self.d) for k in range(self.L + 1)])
        self.projs_i = nn.ModuleList([nn.Linear(self.f, self.d) for k in range(self.L + 1)])
        self.trans_u = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.all_T_u.shape[1], self.all_T_u.shape[1] // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.all_T_u.shape[1] // 2, self.f),
            nn.Dropout(0.5),
        )
        self.trans_i = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.all_T_i.shape[1], self.all_T_i.shape[1] // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.all_T_i.shape[1] // 2, self.f),
            nn.Dropout(0.5),
        )

    @torch.no_grad()
    def _KNN(self, embs, K):
        embs = pca(embs, 150)
        topk_indices = knn(embs, embs, k=K+1)[1].reshape(-1, K + 1)
        return topk_indices[:, 1:].cuda()
    
    def construct_knn_graph(self, all_embed, K, name):
        f_graph = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.T_backbone, self.model_name, f"Graph_{name}_{K}.pkl")
        sucflg, Graph = load_pkls(f_graph)
        if sucflg: return Graph.cuda()
        N = all_embed.shape[0]
        nearestK = self._KNN(all_embed, K)
        row = torch.arange(N).repeat(K, 1).T.reshape(-1).cuda()
        col = nearestK.reshape(-1)
        index = torch.stack([row, col])
        data = torch.ones(index.size(-1)).cuda()
        Graph = torch.sparse_coo_tensor(index, data, torch.Size([N, N]), dtype=torch.float)
        Graph = (Graph + Graph.T) / 2
        Graph = Graph.coalesce()
        Graph = sym_norm_graph(Graph)
        dump_pkls((Graph, f_graph))
        return Graph.cuda()

    def computer(self, mode):
        if mode == "user":
            X = self.trans_u(self.all_T_u)
            self_loop = self_loop_graph(self.num_users).cuda()
            L = self.L_u
            proj_list = self.projs_u
            gamma = self.gamma_u
        else:
            X = self.trans_i(self.all_T_i)
            self_loop = self_loop_graph(self.num_items).cuda()
            L = self.L_i
            proj_list = self.projs_i
            gamma = self.gamma_i

        # scaled Laplacian
        # lambda_max = 2.0
        # index, values, size = L.indices(), L.values(), L.size()
        # values = 2.0 * values / lambda_max
        # L = torch.sparse_coo_tensor(index, values, size, dtype=torch.float)
        # L = L - self_loop

        # filter with Chebyshev polynomial
        X_list, eta_list = [], []
        Tx_0 = X
        h_0 = torch.tanh(proj_list[0](Tx_0))
        gamma_0 = gamma[:,0].unsqueeze(dim=-1)
        eta_0 = torch.matmul(h_0, gamma_0) / self.d
        hidden = torch.matmul(Tx_0.unsqueeze(dim=-1), eta_0.unsqueeze(dim=-1)).squeeze(dim=-1)
        X_list.append(Tx_0)
        eta_list.append(eta_0)
        if self.L == 0:
            return hidden, torch.stack(eta_list,dim=1).squeeze(dim=-1)
        Tx_1 = torch.sparse.mm(L, Tx_0)
        h_1 = torch.tanh(proj_list[1](Tx_1))
        gamma_1 = gamma[:,1].unsqueeze(dim=-1)
        eta_1 = torch.matmul(h_1, gamma_1) / self.d
        hidden = hidden + torch.matmul(Tx_1.unsqueeze(dim=-1), eta_1.unsqueeze(dim=-1)).squeeze(dim=-1)
        X_list.append(Tx_1)
        eta_list.append(eta_1)
        for k in range(1, self.L):
            Tx_2 = 2. * torch.sparse.mm(L, Tx_1) - Tx_0
            Tx_0, Tx_1 = Tx_1, Tx_2
            X_list.append(Tx_1)
            h_k = torch.tanh(proj_list[k + 1](Tx_1))
            gamma_k = gamma[:, k + 1].unsqueeze(dim=-1)
            eta_k = torch.matmul(h_k, gamma_k) / self.d
            hidden = hidden + torch.matmul(Tx_1.unsqueeze(dim=-1), eta_k.unsqueeze(dim=-1)).squeeze(dim=-1)
            eta_list.append(eta_k)
            
        return hidden, torch.stack(eta_list,dim=1).squeeze(dim=-1)
    
    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        all_users, _ = self.computer(mode="user")
        all_items, _ = self.computer(mode="item")
        u = all_users[batch_user]
        i = all_items[batch_pos_item]
        j = all_items[batch_neg_item]
        pos_score = (u * i).sum(dim=1, keepdim=True)
        neg_score = torch.bmm(j, u.unsqueeze(-1)).squeeze(-1)
        pos_score = pos_score.expand_as(neg_score)  # batch_size, num_ns
        loss = -F.logsigmoid(pos_score - neg_score).mean(1).sum()
        return loss
    
    def get_ratings(self, batch_user):
        users, _ = self.computer(mode="user")
        items, _ = self.computer(mode="item")
        users = users[batch_user]
        score_mat = torch.matmul(users, items.T)
        return score_mat


class auxMM(BaseKD4MM):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "auxmm"
        self.K = args.auxmm_K
        self.keep_prob = args.auxmm_keep_prob
        self.dropout_rate = args.auxmm_dropout_rate
        self.hidden_dim_ratio = args.hidden_dim_ratio
        self.aux_params_update_every = args.auxmm_interval
        self.aux_lr = args.aux_lr
        self.global_alpha = args.auxmm_alpha
        self.L = args.auxmm_L
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.all_T_u, self.all_T_i = self.teacher.get_all_embedding()
        
        self.projector_u = Projector(self.student_dim, self.teacher_dim, 1, norm=False, dropout_rate=self.dropout_rate, hidden_dim_ratio=self.hidden_dim_ratio)
        self.projector_i = Projector(self.student_dim, self.teacher_dim, 1, norm=False, dropout_rate=self.dropout_rate, hidden_dim_ratio=self.hidden_dim_ratio)
        self.Graph_u = self.construct_knn_graph(self.all_T_u, self.K, "user")
        self.Graph_i = self.construct_knn_graph(self.all_T_i, self.K, "item")
        self.aux_gates_u = nn.ModuleList([
            nn.Sequential(
            nn.Linear(self.teacher_dim, self.teacher_dim // 2),
            nn.ReLU(),
            nn.Linear(self.teacher_dim // 2, 1),
        ) for _ in range(self.L)])
        self.aux_gates_i = nn.ModuleList([
            nn.Sequential(
            nn.Linear(self.teacher_dim, self.teacher_dim // 2),
            nn.ReLU(),
            nn.Linear(self.teacher_dim // 2, 1),
        ) for _ in range(self.L)])
        self.aux_loader = DataLoader(self.student.dataset, batch_size=self.args.batch_size, shuffle=True)
        aux_base_optimizer = torch.optim.Adam(self.get_aux_params(), lr=self.aux_lr)
        self.aux_optimizer = MetaOptimizer(aux_base_optimizer, hpo_lr=0.1)
        self.global_step, self.aux_step = 0, 0

    def get_main_params(self):
        return [param[1] for param in self.named_parameters() if param[1].requires_grad and "aux" not in param[0]]

    def get_aux_params(self):
        return list(self.aux_gates_u.parameters()) + list(self.aux_gates_i.parameters())
    
    def get_params_to_update(self):
        return [{"params": self.get_main_params(), 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    @torch.no_grad()
    def _KNN(self, embs, K):
        embs = pca(embs, 150)
        topk_indices = knn(embs, embs, k=K+1)[1].reshape(-1, K + 1)
        return topk_indices[:, 1:].cuda()
    
    def construct_knn_graph(self, all_embed, K, name):
        f_graph = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.T_backbone, self.model_name, f"Graph_{name}_{K}.pkl")
        sucflg, Graph = load_pkls(f_graph)
        if sucflg: return Graph.cuda()
        N = all_embed.shape[0]
        nearestK = self._KNN(all_embed, K)
        row = torch.arange(N).repeat(K, 1).T.reshape(-1).cuda()
        col = nearestK.reshape(-1)
        index = torch.stack([row, col])
        data = torch.ones(index.size(-1)).cuda()
        Graph = torch.sparse_coo_tensor(index, data, torch.Size([N, N]), dtype=torch.float)
        Graph = (Graph + Graph.T) / 2
        Graph = Graph.coalesce()
        Graph = sym_norm_graph(Graph)
        dump_pkls((Graph, f_graph))
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
        droped_Graph = torch.sparse_coo_tensor(index.t(), values, size, dtype=torch.float)
        return droped_Graph

    def generate_filter(self, Adj, mode):
        droped_Adj = self._dropout_graph(Adj)
        L = self_loop_graph(droped_Adj.shape[0]).cuda() - droped_Adj
        if mode == "user":
            gates = self.aux_gates_u
            X = self.all_T_u
        else:
            gates = self.aux_gates_i
            X = self.all_T_i
        alphas = []
        for i in range(self.L):
            alpha = torch.sigmoid(gates[i](X).squeeze()) * 2 * self.global_alpha
            alphas.append(alpha)
            torch.save(alpha.detach().cpu(), f"alpha_{mode}_{i}.pt")
        return L, alphas
    
    def conv(self, X, Lap, alphas):
        Z, out = X, X
        for i in range(self.L):
            Z = torch.sparse.mm(Lap, Z)
            out = out - alphas[i].unsqueeze(-1) * Z
        return out
    
    def freq_loss(self, batch_entity, all_S, all_T, adj, alpha, projector):
        proj_all_S = projector(all_S)
        S = self.conv(proj_all_S, adj, alpha)
        T = self.conv(all_T, adj, alpha)
        T_feas = T[batch_entity]
        S_feas = S[batch_entity]
        G_diff = (T_feas - S_feas).pow(2).sum(-1)
        loss = G_diff.sum()
        return loss

    def get_user_loss(self, batch_entity):
        all_T = self.all_T_u
        all_S = self.student.user_emb.weight
        Lap, alpha = self.generate_filter(self.Graph_u, "user")
        loss = self.freq_loss(
            batch_entity, all_S, all_T, 
            Lap, alpha, self.projector_u
            )
        return loss
    
    def get_item_loss(self, batch_entity):
        all_T = self.all_T_i
        all_S_id = self.student.item_emb.weight
        Lap, alpha = self.generate_filter(self.Graph_i, "item")
        loss = self.freq_loss(
            batch_entity, all_S_id, all_T, 
            Lap, alpha, self.projector_i
            )
        return loss

    def cal_KD_loss(self, batch_user, batch_pos_item, batch_neg_item):
        batch_item = torch.cat([batch_pos_item.unique(), batch_neg_item.unique()]).unique()
        loss_user = self.get_user_loss(batch_user.unique())
        loss_item = self.get_item_loss(batch_item)
        loss = loss_user + loss_item
        return loss
    
    def update_aux_params(self, batch_user, batch_pos_item, batch_neg_item):
        if self.global_step % self.aux_params_update_every != 0:
            return
        if self.aux_step % len(self.aux_loader) == 0:
            self.aux_loader.dataset.negative_sampling()
        aux_user, aux_pos_item, aux_neg_item = next(iter(self.aux_loader))
        aux_user, aux_pos_item, aux_neg_item = aux_user.cuda(), aux_pos_item.cuda(), aux_neg_item.cuda()
        output = self.student(batch_user, batch_pos_item, batch_neg_item)
        rec_loss = self.student.get_loss(output)
        kd_loss = self.cal_KD_loss(batch_user, batch_pos_item, batch_neg_item)
        main_loss = rec_loss + self.args.lmbda * kd_loss

        kd_loss_aux = self.cal_KD_loss(aux_user, aux_pos_item, aux_neg_item)
        aux_output = self.student(aux_user, aux_pos_item, aux_neg_item)
        aux_loss = self.student.get_loss(aux_output) + 0. * kd_loss_aux
        self.aux_optimizer.step(
            train_loss=main_loss,
            val_loss=aux_loss,
            aux_params=self.get_aux_params(),
            parameters=self.get_main_params(),
        )
        self.aux_step += 1

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        loss = self.cal_KD_loss(batch_user, batch_pos_item, batch_neg_item)
        self.global_step += 1
        return loss
