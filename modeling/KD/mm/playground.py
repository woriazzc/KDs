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

from ..utils import Projector, pca, load_pkls, dump_pkls, self_loop_graph, sym_norm_graph
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
        self.all_T_mm = self.teacher.get_item_modality_embedding(self.teacher.item_list)
        
        self.projector_u = Projector(self.student_dim, self.teacher_dim, 1, norm=False, dropout_rate=self.dropout_rate, hidden_dim_ratio=self.hidden_dim_ratio)
        self.projector_i = Projector(self.student_dim, self.teacher_dim, 1, norm=False, dropout_rate=self.dropout_rate, hidden_dim_ratio=self.hidden_dim_ratio)
        self.projector_mm = nn.ModuleDict({m: Projector(self.student_dim, self.teacher_dim, 1, norm=False, dropout_rate=self.dropout_rate, hidden_dim_ratio=self.hidden_dim_ratio)
                                            for m in self.all_T_mm})
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
        rndS = random.random() * alpha * 2
        rndT = random.random() * alpha * 2
        self_loop = self_loop_graph(droped_Adj.shape[0]).cuda()
        H_S = droped_Adj * rndS + self_loop * (1. - rndS)
        H_T = droped_Adj * rndT + self_loop * (1. - rndT)
        return H_S, H_T
    
    def freq_loss(self, batch_entity, all_S, all_T, GraphS, GraphT, projector):
        S = torch.sparse.mm(GraphS, all_S)
        T = torch.sparse.mm(GraphT, all_T)
        T_feas = T[batch_entity]
        S_feas = S[batch_entity]
        S_feas = projector(S_feas)
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
        all_S = self.student.user_emb.weight
        all_T = self.all_T_u
        projector = self.projector_u
        H_S_u, H_T_u = self.generate_filter(self.Graph_u, self.alpha)
        loss = self.freq_loss(batch_entity, all_S, all_T, H_S_u, H_T_u, projector)
        return loss
    
    def get_item_loss(self, batch_entity):
        all_T = self.all_T_i
        all_S_id = self.student.item_emb.weight
        H_S_i, H_T_i = self.generate_filter(self.Graph_i, self.alpha)
        loss = self.freq_loss(batch_entity, all_S_id, all_T, H_S_i, H_T_i, self.projector_i)
        return loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        batch_item = torch.cat([batch_pos_item.unique(), batch_neg_item.unique()]).unique()
        loss_user = self.get_user_loss(batch_user.unique())
        loss_item = self.get_item_loss(batch_item)
        loss = loss_user + loss_item
        return loss


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
