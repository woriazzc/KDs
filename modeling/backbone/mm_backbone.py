import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseMM
from .base_layer import BipartitleGraph
from .utils import EmbLoss


class BM3(BaseMM):
    def __init__(self, dataset, mm_dict, args):
        super().__init__(dataset, mm_dict, args)
        self.embedding_dim = args.embedding_dim
        self.feat_embed_dim = args.embedding_dim
        self.n_layers = args.n_layers
        self.reg_weight = args.reg_weight
        self.cl_weight = args.cl_weight
        self.dropout = args.dropout

        self.modality_names = list(mm_dict.keys())
        self.Graph = BipartitleGraph(args, dataset).graph
        self.user_id_emb = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_id_emb = nn.Embedding(self.num_items, self.embedding_dim)
        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()
        self.mm_emb_dict = nn.ModuleDict({
            name: nn.Embedding.from_pretrained(data, freeze=False) for (name, data) in mm_dict.items()
        })
        self.mm_trs_dict = nn.ModuleDict({
            name: nn.Linear(data.shape[1], self.feat_embed_dim) for (name, data) in mm_dict.items()
        })
        self.reset_para()

    def reset_para(self):
        nn.init.xavier_uniform_(self.user_id_emb.weight)
        nn.init.xavier_uniform_(self.item_id_emb.weight)
        nn.init.xavier_normal_(self.predictor.weight)
        for m in self.mm_trs_dict:
            nn.init.xavier_normal_(self.mm_trs_dict[m].weight)
    
    def computer(self):
        h = self.item_id_emb.weight
        ego_embeddings = torch.cat((self.user_id_emb.weight, self.item_id_emb.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.Graph, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h
    
    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        u_online_ori, i_online_ori = self.computer()
        u_online_ori = u_online_ori[batch_user]
        i_online_ori = i_online_ori[batch_pos_item]
        mm_onlines, mm_targets = {}, {}
        for m in self.mm_emb_dict:
            trs = self.mm_trs_dict[m]
            feat = self.mm_emb_dict[m](batch_pos_item)
            mm_onlines[m] = trs(feat)
        with torch.no_grad():
            u_target, i_target = u_online_ori.clone().detach(), i_online_ori.clone().detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)
            for m in mm_onlines:
                mm_target = mm_onlines[m].clone()
                mm_targets[m] = F.dropout(mm_target, self.dropout)
        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)
        loss_mi, loss_mm = 0., 0.
        for m in mm_onlines:
            mm_online = self.predictor(mm_onlines[m])
            mm_target = mm_targets[m]
            loss_mi += 1. - F.cosine_similarity(mm_online, i_target.detach(), dim=-1).mean()
            loss_mm += 1. - F.cosine_similarity(mm_online, mm_target.detach(), dim=-1).mean()
        loss_ui = 1. - F.cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1. - F.cosine_similarity(i_online, u_target.detach(), dim=-1).mean()
        loss_reg = self.reg_loss(u_online_ori, i_online_ori)
        return loss_ui, loss_iu, loss_mi, loss_mm, loss_reg
    
    def get_loss(self, output):
        loss_ui, loss_iu, loss_mi, loss_mm, loss_reg = output
        loss = (loss_ui + loss_iu) + self.reg_weight * loss_reg + \
               self.cl_weight * (loss_mi + loss_mm)
        return loss
    
    def get_user_embedding(self, batch_user):
        all_users, all_items = self.computer()
        users = self.predictor(all_users[batch_user])
        return users
    
    def get_item_embedding(self, batch_item):
        all_users, all_items = self.computer()
        items = self.predictor(all_items[batch_item])
        return items

    def get_all_embedding(self):
        all_users, all_items = self.computer()
        users = self.predictor(all_users)
        items = self.predictor(all_items)
        return users, items
    
    def get_item_modality_embedding(self, batch_item):
        mm_feat_dict = {}
        for modality in self.mm_emb_dict:
            trs = self.mm_trs_dict[modality]
            feat = self.mm_emb_dict[modality](batch_item)
            mm_feat_dict[modality] = trs(feat)
        return mm_feat_dict
    
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
        all_users, all_items = self.computer()
        
        u = self.predictor(all_users[batch_user])		# batch_size x dim
        i = self.predictor(all_items[batch_items])		# batch_size x k x dim
        
        score = torch.bmm(i, u.unsqueeze(-1)).squeeze(-1)   # batch_size, k
        
        return score
    
    def get_all_ratings(self):
        users, items = self.get_all_embedding()
        score_mat = torch.matmul(users, items.T)
        return score_mat

    def get_ratings(self, batch_user):
        users, items = self.get_all_embedding()
        users = users[batch_user]
        score_mat = torch.matmul(users, items.T)
        return score_mat
