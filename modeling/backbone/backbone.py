import os
import math
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseRec, BaseGCN


class BPR(BaseRec):
    def __init__(self, dataset, args):
        """
        Parameters
        ----------
        num_users : int
        num_users : int
        dim : int
            embedding dimension
        """
        super(BPR, self).__init__(dataset, args)

        self.init_std = args.init_std
        self.embedding_dim = args.embedding_dim

        # User / Item Embedding
        self.user_emb = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_emb = nn.Embedding(self.num_items, self.embedding_dim)

        self.reset_para()

    def reset_para(self):
        nn.init.normal_(self.user_emb.weight, mean=0., std=self.init_std)
        nn.init.normal_(self.item_emb.weight, mean=0., std=self.init_std)
        
    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        """
        Parameters
        ----------
        batch_user : 1-D LongTensor (batch_size)
        batch_pos_item : 1-D LongTensor (batch_size)
        batch_neg_item : 1-D LongTensor (batch_size)

        Returns
        -------
        output : 
            Model output to calculate its loss function
        """
        
        u = self.user_emb(batch_user)
        i = self.item_emb(batch_pos_item)
        j = self.item_emb(batch_neg_item)
        
        pos_score = (u * i).sum(dim=1, keepdim=True)
        neg_score = (u * j).sum(dim=1, keepdim=True)

        return pos_score, neg_score

    def forward_multi_items(self, batch_user, batch_items):
        """forward when we have multiple items for a user

        Parameters
        ----------
        batch_user : 1-D LongTensor (batch_size)
        batch_items : 2-D LongTensor (batch_size x k)

        Returns
        -------
        score : 2-D FloatTensor (batch_size x k)
        """

        batch_user = batch_user.unsqueeze(-1)
        batch_user = torch.cat(batch_items.size(1) * [batch_user], 1)
        
        u = self.user_emb(batch_user)		# batch_size x k x dim
        i = self.item_emb(batch_items)		# batch_size x k x dim
        
        score = (u * i).sum(dim=-1, keepdim=False)
        
        return score

    def get_user_embedding(self, batch_user):
        return self.user_emb(batch_user)
    
    def get_item_embedding(self, batch_item):
        return self.item_emb(batch_item)

    def get_all_pre_embedding(self):
        """get total embedding of users and items

        Returns
        -------
        users : 2-D FloatTensor (num. users x dim)
        items : 2-D FloatTensor (num. items x dim)
        """
        users = self.user_emb(self.user_list)
        items = self.item_emb(self.item_list)

        return users, items
    
    def get_all_ratings(self):
        users, items = self.get_all_pre_embedding()
        score_mat = torch.matmul(users, items.T)
        return score_mat
    
    def get_ratings(self, batch_user):
        users, items = self.get_all_pre_embedding()
        users = users[batch_user]
        score_mat = torch.matmul(users, items.T)
        return score_mat


class LightGCN(BaseGCN):
    def __init__(self, dataset, args):
        super(LightGCN, self).__init__(dataset, args)
        
        self.embedding_dim = args.embedding_dim
        self.num_layers = args.num_layers
        self.keep_prob = getattr(args, "keep_prob", 0.)
        self.A_split = getattr(args, "A_split", False)
        self.dropout = getattr(args, "dropout", False)
        self.init_std = args.init_std

        self.user_emb = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.item_emb = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
        
        self.Graph = self.dataset.SparseGraph

        self.aug_graph = getattr(args, "aug_graph", False)
        if self.aug_graph:
            self.augGraph = pickle.load(open(os.path.join("data", args.dataset, "aug_graph.pkl"), "rb")).cuda()
        else:
            self.augGraph = None

        self.reset_para()
    
    def reset_para(self):
        nn.init.normal_(self.user_emb.weight, std=self.init_std)
        nn.init.normal_(self.item_emb.weight, std=self.init_std)

    def _dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse_coo_tensor(index.t(), values, size, dtype=torch.float32)
        return g.coalesce()

    def _dropout(self, keep_prob, Graph):
        if self.A_split:
            graph = []
            for g in Graph:
                graph.append(self._dropout_x(g, keep_prob))
        else:
            graph = self._dropout_x(Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        if self.training and self.aug_graph:
            Graph = self.augGraph
        else:
            Graph = self.Graph
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        all_emb = torch.cat([users_emb, items_emb])
        light_out = all_emb
        if self.dropout:
            if self.training:
                g_droped = self._dropout(self.keep_prob, Graph)
            else:
                g_droped = Graph
        else:
            g_droped = Graph

        for layer in range(1, self.num_layers + 1):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            light_out = (light_out * layer + all_emb) / (layer + 1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def get_all_pre_embedding(self):
        users = self.user_emb(self.user_list)
        items = self.item_emb(self.item_list)
        
        return users, items


# based on https://github.com/liu-jc/PyTorch_NGCF/blob/master/NGCF/Models.py
class NGCF(LightGCN):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()
        self.weight_size = [self.embedding_dim] * self.num_layers
        self.weight_size = [self.embedding_dim] + self.weight_size
        dropout_list = [args.dropout_rate] * self.num_layers
        for i in range(self.num_layers):
            self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
            self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
            self.dropout_list.append(nn.Dropout(dropout_list[i]))

    def computer(self):
        """
        propagate methods for NGCF
        """
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        ego_embeddings = torch.cat([users_emb, items_emb])
        ngcf_out = [ego_embeddings]
        if self.dropout:
            if self.training:
                g_droped = self._dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        
        for i in range(self.num_layers):
            side_embeddings = torch.sparse.mm(g_droped, ego_embeddings)
            sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = self.dropout_list[i](ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            ngcf_out += [norm_embeddings]
        
        all_embeddings = torch.cat(ngcf_out, dim=1)
        users, items = torch.split(all_embeddings, [self.num_users, self.num_items])
        return users, items
