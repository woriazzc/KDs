import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseCF, BaseGCN
from .base_layer import BehaviorAggregator


class Prediction(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.model_type = "preload"
        self.model_name = "preload"
        self.score_mat = torch.zeros((dataset.num_users, dataset.num_items))
    
    def load_state_dict(self, score_mat: torch.Tensor):
        assert score_mat.shape == self.score_mat.shape
        self.score_mat = score_mat

    def get_all_ratings(self):
        return self.score_mat.cuda()
    
    def get_ratings(self, batch_user):
        return self.score_mat[batch_user.cpu()].cuda()
    
    def forward_multi_items(self, batch_user, batch_items):
        return self.score_mat[batch_user.unsqueeze(-1).cpu(), batch_items.cpu()].cuda()


class BPR(BaseCF):
    def __init__(self, dataset, args):
        super(BPR, self).__init__(dataset, args)
        self.model_name = "bpr"
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
        batch_neg_item : 2-D LongTensor (batch_size, num_ns)

        Returns
        -------
        output : 
            Model output to calculate its loss function
        """
        
        u = self.user_emb(batch_user)       # batch_size, dim
        i = self.item_emb(batch_pos_item)   # batch_size, dim
        j = self.item_emb(batch_neg_item)   # batch_size, num_ns, dim
        
        pos_score = (u * i).sum(dim=1, keepdim=True)    # batch_size, 1
        neg_score = torch.bmm(j, u.unsqueeze(-1)).squeeze(-1)       # batch_size, num_ns

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
        
        u = self.user_emb(batch_user)		# batch_size x dim
        i = self.item_emb(batch_items)		# batch_size x k x dim
        
        score = torch.bmm(i, u.unsqueeze(-1)).squeeze(-1)   # batch_size, k
        
        return score

    def get_user_embedding(self, batch_user):
        return self.user_emb(batch_user)
    
    def get_item_embedding(self, batch_item):
        return self.item_emb(batch_item)

    def get_all_embedding(self):
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
        users, items = self.get_all_embedding()
        score_mat = torch.matmul(users, items.T)
        return score_mat
    
    def get_ratings(self, batch_user):
        users = self.get_user_embedding(batch_user.cuda())
        items = self.get_item_embedding(self.item_list)
        score_mat = torch.matmul(users, items.T)
        return score_mat


class LightGCN(BaseGCN):
    def __init__(self, dataset, args):
        super(LightGCN, self).__init__(dataset, args)
        self.model_name = "lightgcn"
        self.embedding_dim = args.embedding_dim
        self.num_layers = args.num_layers
        self.keep_prob = getattr(args, "keep_prob", 0.)
        self.A_split = getattr(args, "A_split", False)
        self.dropout = getattr(args, "dropout", False)
        self.init_std = args.init_std

        self.user_emb = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.item_emb = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)

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
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        all_emb = torch.cat([users_emb, items_emb])
        light_out = all_emb
        if self.dropout:
            if self.training:
                g_droped = self._dropout(self.keep_prob, self.Graph)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

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


# Refer to https://github.com/wu1hong/SCCF/blob/master/recbole/model/general_recommender/sccf.py
def xavier_normal_initialization(module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

class SCCF(BaseCF):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.model_name = "sccf"
        self.embedding_dim = args.embedding_dim
        self.temperature = args.sccf_temperature
        self.user_emb = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_emb = nn.Embedding(self.num_items, self.embedding_dim)
        self.apply(xavier_normal_initialization)

    def encode_with_norm(self, batch_user, batch_item):
        user_e = self.user_emb(batch_user)
        item_e = self.item_emb(batch_item)
        return F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)

    def forward(self, batch_user, batch_item, _):
        """
        Parameters
        ----------
        batch_user : 1-D LongTensor (batch_size)
        batch_item : 1-D LongTensor (batch_size)
        _ : 2-D LongTensor (batch_size, num_ns), batch_negative_items, not used in this model

        Returns
        -------
        output : 
            Model output to calculate its loss function
        """
        u_idx, u_inv_idx, u_counts = torch.unique(batch_user, return_counts=True, return_inverse=True)
        i_idx, i_inv_idx, i_counts = torch.unique(batch_item, return_counts=True, return_inverse=True)
        u_counts, i_counts = u_counts.reshape(-1, 1).float(), i_counts.reshape(-1, 1).float()
        user_e, item_e = self.encode_with_norm(batch_user, batch_item)
        ip = (user_e * item_e).sum(dim=1)
        up_score = (ip / self.temperature).exp() + (ip ** 2 / self.temperature).exp()
        up = up_score.log().mean()
        user_e, item_e = self.encode_with_norm(u_idx, i_idx)
        sim_mat = user_e @ item_e.T
        score = (sim_mat / self.temperature).exp() + (sim_mat ** 2 / self.temperature).exp()
        down = (score * (u_counts @ i_counts.T)).mean().log()
        return up, down
    
    def get_loss(self, output):
        """Compute the loss function with the model output

        Parameters
        ----------
        output : 
            model output (results of forward function)

        Returns
        -------
        loss : float
        """
        up, down = output
        loss = -up + down
        return loss

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
        
        u = self.user_emb(batch_user)		# batch_size x dim
        i = self.item_emb(batch_items)		# batch_size x k x dim
        score = torch.bmm(i, u.unsqueeze(-1)).squeeze(-1)   # batch_size, k
        return score

    def get_user_embedding(self, batch_user):
        return self.user_emb(batch_user)
    
    def get_item_embedding(self, batch_item):
        return self.item_emb(batch_item)

    def get_all_embedding(self):
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
        users, items = self.get_all_embedding()
        score_mat = torch.matmul(users, items.T)
        return score_mat
    
    def get_ratings(self, batch_user):
        users = self.get_user_embedding(batch_user.cuda())
        items = self.get_item_embedding(self.item_list)
        score_mat = torch.matmul(users, items.T)
        return score_mat


# Refer to https://github.com/reczoo/RecZoo/blob/main/matching/cf/SimpleX/src/SimpleX.py
class SimpleX(BaseCF):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.model_name = "simplex"
        self.init_std = args.init_std
        self.embedding_dim = args.embedding_dim
        self.aggregator = "mean"    # Only support arrgregator='mean', which performs best in the paper
        self.dropout_rate = args.simplex_dropout_rate
        self.similarity_score = args.simplex_similarity_score
        self.gamma = args.simplex_gamma
        self.margin = args.simplex_margin
        self.max_len = args.simplex_max_len
        self.negative_weight = args.simplex_negative_weight
        self.enable_bias = args.simplex_enable_bias
        self.pad_id = self.num_items

        self.user_emb = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_emb = nn.Embedding(self.num_items + 1, self.embedding_dim, padding_idx=self.pad_id)

        self.user_history = self.create_user_history()
        self.behavior_aggregation = BehaviorAggregator(self.embedding_dim, gamma=self.gamma, aggregator=self.aggregator)

        if self.enable_bias:
            self.user_bias = nn.Embedding(self.num_users, 1)
            self.item_bias = nn.Embedding(self.num_items + 1, 1, padding_idx=self.pad_id)
            self.global_bias = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(self.dropout_rate)

        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Embedding:
                if m.padding_idx is not None:
                    # using the last index as padding_idx
                    assert m.padding_idx == m.weight.shape[0] - 1
                    nn.init.normal_(m.weight[0:-1, :], mean=0., std=self.init_std)
                else:
                    nn.init.normal_(m.weight, mean=0., std=self.init_std)

    def create_user_history(self):
        train_dict = self.dataset.train_dict
        user_history = []
        for u in self.user_list:
            history = train_dict[u.item()][-self.max_len:]
            if len(history) < self.max_len: history = torch.cat([torch.ones(self.max_len - len(history), dtype=history.dtype) * self.pad_id, history], dim=0)
            user_history.append(history)
        user_history = torch.stack(user_history, dim=0)
        return user_history.cuda()
        
    def user_tower(self, batch_user):
        uid_emb = self.user_emb(batch_user)
        user_history_emb = self.item_emb(self.user_history[batch_user])
        user_vec = self.behavior_aggregation(uid_emb, user_history_emb)
        if self.similarity_score == "cosine":
            user_vec = F.normalize(user_vec, dim=-1)
        if self.enable_bias:
            user_vec = torch.cat([user_vec, torch.ones(user_vec.size(0), 1).to(user_vec.device)], dim=-1)
        user_vec = self.dropout(user_vec)
        return user_vec

    def item_tower(self, batch_item):
        item_vec = self.item_emb(batch_item)
        if self.similarity_score == "cosine":
            item_vec = F.normalize(item_vec, dim=-1)
        if self.enable_bias:
            item_vec = torch.cat([item_vec, self.item_bias(batch_item)], dim=-1)
        return item_vec

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        """
        Parameters
        ----------
        batch_user : 1-D LongTensor (batch_size)
        batch_pos_item : 1-D LongTensor (batch_size)
        batch_neg_item : 2-D LongTensor (batch_size, num_ns)

        Returns
        -------
        output : 
            Model output to calculate its loss function
        """
        
        u = self.user_tower(batch_user)
        i = self.item_tower(batch_pos_item)
        j = self.item_tower(batch_neg_item)
        
        pos_score = (u * i).sum(dim=1, keepdim=True)    # batch_size, 1
        neg_score = torch.bmm(j, u.unsqueeze(-1)).squeeze(-1)       # batch_size, num_ns
        if self.enable_bias: # user_bias and global_bias only influence training, but not inference for ranking
            pos_score += self.user_bias(batch_user) + self.global_bias
            neg_score += self.user_bias(batch_user) + self.global_bias
        return pos_score, neg_score
    
    def get_loss(self, output):
        pos_score, neg_score = output[0], output[1]
        pos_loss = torch.relu(1. - pos_score)
        neg_loss = torch.relu(neg_score - self.margin)
        loss = pos_loss + neg_loss.mean(dim=-1) * self.negative_weight
        return loss.mean()

    def forward_multi_items(self, batch_user, batch_items):
        u = self.user_tower(batch_user)		# batch_size x dim
        i = self.item_tower(batch_items)		# batch_size x k x dim
        
        score = torch.bmm(i, u.unsqueeze(-1)).squeeze(-1)   # batch_size, k
        if self.enable_bias:
            score += self.user_bias(batch_user) + self.global_bias
        return score

    def get_user_embedding(self, batch_user):
        return self.user_tower(batch_user)
    
    def get_item_embedding(self, batch_item):
        return self.item_tower(batch_item)

    def get_all_embedding(self):
        # To save CUDA memory
        users = torch.zeros_like(self.user_emb.weight.data)
        batch_size = self.args.batch_size
        n_batch = math.ceil(self.num_users / batch_size)
        for i in range(n_batch):
            users[i * batch_size:(i + 1) * batch_size] = self.user_tower(self.user_list[i * batch_size:(i + 1) * batch_size])
        items = self.item_tower(self.item_list)
        return users, items
    
    def get_all_ratings(self):
        users, items = self.get_all_embedding()
        score_mat = torch.matmul(users, items.T)
        if self.enable_bias:
            score_mat += self.user_bias(self.user_list) + self.global_bias
        return score_mat
    
    def get_ratings(self, batch_user):
        users = self.get_user_embedding(batch_user.cuda())
        items = self.get_item_embedding(self.item_list)
        score_mat = torch.matmul(users, items.T)
        if self.enable_bias:
            score_mat += self.user_bias(batch_user.cuda()) + self.global_bias
        return score_mat


# Refer to https://github.com/Coder-Yu/SELFRec/blob/main/model/graph/XSimGCL.py
class XSimGCL(BaseGCN):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.model_name = "xsimgcl"
        self.embedding_dim = args.embedding_dim
        self.num_layers = args.num_layers
        self.layer_cl = args.layer_cl
        self.eps = args.eps
        self.w_cl = args.w_cl
        self.tau = args.tau

        self.user_emb = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.item_emb = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)

        self.reset_para()
    
    def reset_para(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def computer(self, perturbed=False):
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        all_emb = torch.cat([users_emb, items_emb])
        gcn_out = torch.zeros_like(all_emb)
        perturbed_out = None

        for layer in range(self.num_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            if perturbed:
                random_noise = torch.rand_like(all_emb).cuda()
                all_emb += torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * self.eps
            gcn_out = (gcn_out * layer + all_emb) / (layer + 1)
            if layer == self.layer_cl - 1:
                perturbed_out = all_emb
        users, items = torch.split(gcn_out, [self.num_users, self.num_items])
        if perturbed:
            perturbed_users, perturbed_items = torch.split(perturbed_out, [self.num_users, self.num_items])
            return users, items, perturbed_users, perturbed_items
        else:
            return users, items

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        all_users, all_items, perturbed_users, perturbed_items = self.computer(True)

        u = all_users[batch_user]
        i = all_items[batch_pos_item]
        j = all_items[batch_neg_item]

        uid, iid = batch_user.unique(), batch_pos_item.unique()
        ori_u, ori_i = all_users[uid], all_items[iid]
        pert_u, pert_i = perturbed_users[uid], perturbed_items[iid]     # batch_size, embed_dim
        
        pos_score = (u * i).sum(dim=1, keepdim=True)    # batch_size, 1
        neg_score = torch.bmm(j, u.unsqueeze(-1)).squeeze(-1)       # batch_size, num_ns

        return pos_score, neg_score, ori_u, ori_i, pert_u, pert_i
    
    def InfoNCE(self, view1, view2, temperature: float, b_cos: bool = True):
        """
        Args:
            view1: (torch.Tensor - N x D)
            view2: (torch.Tensor - N x D)
            temperature: float
            b_cos (bool)
        Return: Average InfoNCE Loss
        """
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()

    def get_loss(self, output):
        pos_score, neg_score, ori_u, ori_i, pert_u, pert_i = output
        pos_score = pos_score.expand_as(neg_score)  # batch_size, num_ns
        bpr_loss = -F.logsigmoid(pos_score - neg_score).mean(1).sum()
        cl_loss = self.InfoNCE(ori_u, pert_u, self.tau) + self.InfoNCE(ori_i, pert_i, self.tau)
        loss = bpr_loss + self.w_cl * cl_loss
        return loss
