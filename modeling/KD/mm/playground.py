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

        self.alpha1 = args.freqmm_alpha1
        self.alpha2 = args.freqmm_alpha2
        self.K = args.freqmm_K
        self.beta = args.freqmm_beta
        self.keep_prob = args.freqmm_keep_prob
        self.dropout_rate = args.freqmm_dropout_rate
        self.hidden_dim_ratio = args.hidden_dim_ratio
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim
        self.modality_names = self.teacher.modality_names

        self.projector_u = Projector(self.student_dim, self.teacher_dim, 1, norm=True, dropout_rate=self.dropout_rate, hidden_dim_ratio=self.hidden_dim_ratio)
        self.projector_i = Projector(self.student_dim, self.teacher_dim, 1, norm=True, dropout_rate=self.dropout_rate, hidden_dim_ratio=self.hidden_dim_ratio)
        self.projector_mm = nn.ModuleDict({m: Projector(self.student_dim, self.teacher_dim, 1, norm=True, dropout_rate=self.dropout_rate, hidden_dim_ratio=self.hidden_dim_ratio)
                                            for m in self.modality_names})

        all_u, all_i = self.teacher.get_all_embedding()
        self.Graph_u = self.construct_knn_graph(all_u, self.K, "user")
        self.Graph_i = self.construct_knn_graph(all_i, self.K, "item")
        self.all_T_u, self.all_T_i = self.teacher.get_all_embedding()
        self.all_T_mm = self.teacher.get_item_modality_embedding(self.teacher.item_list)
    
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
        T_feas = F.normalize(T_feas, p=2, dim=-1)
        cos_theta = (T_feas * S_feas).sum(-1, keepdim=True)
        G_diff = 1. - cos_theta
        loss = G_diff.sum()
        return loss

    def get_user_loss(self, batch_entity):
        all_S = self.student.user_emb.weight
        all_T = self.all_T_u
        projector = self.projector_u
        H_S_u, H_T_u = self.generate_filter(self.Graph_u, self.alpha1)
        loss = self.freq_loss(batch_entity, all_S, all_T, H_S_u, H_T_u, projector)
        return loss
    
    def get_item_loss(self, batch_entity):
        all_T = self.all_T_i
        all_S_id = self.student.item_emb.weight
        H_S_i, H_T_i = self.generate_filter(self.Graph_i, self.alpha1)
        loss = self.freq_loss(batch_entity, all_S_id, all_T, H_S_i, H_T_i, self.projector_i)
        for m in self.all_T_mm:
            H_S_mm, H_T_mm = self.generate_filter(self.Graph_i, self.alpha2)
            loss_m = self.freq_loss(batch_entity, all_S_id, self.all_T_mm[m], H_S_mm, H_T_mm, self.projector_mm[m])
            loss += self.beta * loss_m
        return loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        batch_item = torch.cat([batch_pos_item.unique(), batch_neg_item.unique()]).unique()
        loss_user = self.get_user_loss(batch_user.unique())
        loss_item = self.get_item_loss(batch_item)
        loss = loss_user + loss_item
        return loss
