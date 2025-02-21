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

from ..utils import Projector, pca, load_pkls, dump_pkls, self_loop_graph
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
        self.nearestK_u, self.nearestK_i = self.get_nearest_K(all_u, all_i, self.K)
        self.Graph_u = self.construct_knn_graph(self.nearestK_u)
        self.Graph_i = self.construct_knn_graph(self.nearestK_i)
        self.all_T_u, self.all_T_i = self.teacher.get_all_embedding()
        self.all_T_mm = self.teacher.get_item_modality_embedding(self.teacher.item_list)
    
    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def _KNN(self, embs, K):
        with torch.no_grad():
            embs = pca(embs, 150)
            topk_indices = knn(embs, embs, k=K+1)[1].reshape(-1, K + 1)
        return topk_indices[:, 1:].cuda()

    def get_nearest_K(self, all_u, all_i, K):
        f_nearestK_u = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.T_backbone, self.model_name, f"nearest_{K}_u.pkl")
        f_nearestK_i = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.T_backbone, self.model_name, f"nearest_{K}_i.pkl")
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
        Graph = torch.sparse_coo_tensor(index, data, torch.Size([N, N]), dtype=torch.float)
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

        rndS = random.random() * self.alpha1 * 2
        rndT = random.random() * self.alpha1 * 2
        valuesS = torch.cat([values * rndS, self_loop_data * (1. - rndS)])
        valuesT = torch.cat([values * rndT, self_loop_data * (1. - rndT)])

        index = torch.cat([index, self_loop_idx], dim=0)

        droped_GraphS = torch.sparse_coo_tensor(index.t(), valuesS, size, dtype=torch.float)
        droped_GraphT = torch.sparse_coo_tensor(index.t(), valuesT, size, dtype=torch.float)
        return droped_GraphS, droped_GraphT
    
    def _dropout_graph_mm(self, Graph):
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

        rndS = random.random() * self.alpha2 * 2
        rndT = random.random() * self.alpha2 * 2
        # valuesS = torch.cat([values * (-rndS), self_loop_data * (1. - rndS)])
        # valuesT = torch.cat([values * (-rndT), self_loop_data * (1. - rndT)])
        valuesS = torch.cat([values * rndS, self_loop_data * (1. - rndS)])
        valuesT = torch.cat([values * rndT, self_loop_data * (1. - rndT)])

        index = torch.cat([index, self_loop_idx], dim=0)

        droped_GraphS = torch.sparse_coo_tensor(index.t(), valuesS, size, dtype=torch.float)
        droped_GraphT = torch.sparse_coo_tensor(index.t(), valuesT, size, dtype=torch.float)
        return droped_GraphS, droped_GraphT

    def freq_loss(self, batch_entity, all_S, all_T, GraphS, GraphT, projector):
        S = torch.sparse.mm(GraphS, all_S)
        T = torch.sparse.mm(GraphT, all_T)
        T_feas = T[batch_entity]
        S_feas = S[batch_entity]
        S_feas = projector(S_feas)
        norm_T = T_feas.pow(2).sum(-1, keepdim=True).pow(1. / 2)
        T_feas = T_feas.div(norm_T)
        cos_theta = (T_feas * S_feas).sum(-1, keepdim=True)
        G_diff = 1. - cos_theta
        loss = G_diff.sum()
        return loss

    def get_user_loss(self, batch_entity):
        all_S = self.student.user_emb.weight
        all_T = self.all_T_u
        projector = self.projector_u
        droped_GraphS, droped_GraphT = self._dropout_graph(self.Graph_u)
        loss = self.freq_loss(batch_entity, all_S, all_T, droped_GraphS, droped_GraphT, projector)
        return loss
    
    def get_item_loss(self, batch_entity):
        droped_GraphS, droped_GraphT = self._dropout_graph(self.Graph_i)
        all_T = self.all_T_i
        all_S_id = self.student.item_emb.weight
        loss = self.freq_loss(batch_entity, all_S_id, all_T, droped_GraphS, droped_GraphT, self.projector_i)
        droped_GraphS, droped_GraphT = self._dropout_graph_mm(self.Graph_i)
        for m in self.all_T_mm:
            loss_m = self.freq_loss(batch_entity, all_S_id, self.all_T_mm[m], droped_GraphS, droped_GraphT, self.projector_mm[m])
            loss += self.beta * loss_m
        return loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        batch_item = torch.cat([batch_pos_item.unique(), batch_neg_item.unique()]).unique()
        loss_user = self.get_user_loss(batch_user.unique())
        loss_item = self.get_item_loss(batch_item)
        loss = loss_user + loss_item
        return loss
