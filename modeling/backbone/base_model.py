import os
import yaml
import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_layer import Embedding
from .utils import convert_sp_mat_to_sp_tensor, load_pkls, dump_pkls


class BaseCF(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        
        self.model_type = "cf"
        self.dataset = dataset
        self.args = args

        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items

        self.user_list = torch.LongTensor([i for i in range(self.num_users)]).cuda()
        self.item_list = torch.LongTensor([i for i in range(self.num_items)]).cuda()
    
    def forward(self):
        raise NotImplementedError
    
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
        raise NotImplementedError
    
    def get_all_ratings(self):
        raise NotImplementedError

    def get_ratings(self, batch_user):
        raise NotImplementedError
    
    def get_all_embedding(self):
        raise NotImplementedError
    
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
        pos_score, neg_score = output[0], output[1]
        pos_score = pos_score.expand_as(neg_score)  # batch_size, num_ns
        loss = -F.logsigmoid(pos_score - neg_score).mean(1).sum()
        return loss
    
    def score_mat(self):
        return self.get_all_ratings()


class BaseSR(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        
        self.model_type = "sr"
        self.dataset = dataset
        self.args = args

        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items

        self.user_list = torch.LongTensor([i + 1 for i in range(self.num_users)]).cuda()
        self.item_list = torch.LongTensor([i + 1 for i in range(self.num_items)]).cuda()
    
    def forward(self):
        raise NotImplementedError
    
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
        raise NotImplementedError
    
    def get_ratings(self, batch_user):
        raise NotImplementedError
    
    def get_all_ratings(self):
        """
        get score mat including all users
        excludes item 0
        """
        batch_size = 1024
        num_batch = math.ceil(self.num_users / batch_size)
        all_scores = []
        for i in range(num_batch):
            batch_user = torch.arange(i * batch_size, min(self.num_users, (i + 1) * batch_size), dtype=torch.long).cuda()
            all_scores.append(self.get_ratings(batch_user))
        all_scores = torch.cat(all_scores, dim=0)
        return all_scores
    

class BaseGCN(BaseCF):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)

    def _construct_small_graph(self):
        user_dim = torch.LongTensor(self.dataset.train_pairs[:, 0].cpu())
        item_dim = torch.LongTensor(self.dataset.train_pairs[:, 1].cpu())

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
        return Graph
    
    def _construct_large_graph(self):
        adj_mat = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        train_pairs = self.dataset.train_pairs.numpy()
        UserItemNet  = csr_matrix((np.ones(len(train_pairs)), (train_pairs[:, 0], train_pairs[:, 1])), shape=(self.num_users, self.num_items))
        R = UserItemNet.tolil()
        adj_mat[:self.num_users, self.num_users:] = R
        adj_mat[self.num_users:, :self.num_users] = R.T
        adj_mat = adj_mat.todok()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        Graph = convert_sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce()
        return Graph

    def construct_graph(self):
        f_Graph = os.path.join("modeling", "backbone", "crafts", self.args.dataset, f"Graph.pkl")
        sucflg, Graph = load_pkls(f_Graph)
        if sucflg:
            return Graph.cuda()
        
        config = yaml.load(open(os.path.join(self.args.DATA_DIR, self.args.dataset, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
        if "large" in config and config["large"] == True:
            Graph = self._construct_large_graph()
        else:
            Graph = self._construct_small_graph()
        dump_pkls((Graph, f_Graph))
        return Graph.cuda()
    
    def computer(self):
        """
        propagate methods
        """
        raise NotImplementedError

    def get_all_pre_embedding(self):
        """get total embedding of users and items before convolution

        Returns
        -------
        users : 2-D FloatTensor (num. users x dim)
        items : 2-D FloatTensor (num. items x dim)
        """
        users = self.user_emb(self.user_list)
        items = self.item_emb(self.item_list)
        
        return users, items
    
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
        all_users, all_items = self.computer()

        u = all_users[batch_user]
        i = all_items[batch_pos_item]
        j = all_items[batch_neg_item]
        
        pos_score = (u * i).sum(dim=1, keepdim=True)    # batch_size, 1
        neg_score = torch.bmm(j, u.unsqueeze(-1)).squeeze(-1)       # batch_size, num_ns

        return pos_score, neg_score
    
    def get_user_embedding(self, batch_user):
        all_users, all_items = self.computer()
        users = all_users[batch_user]
        return users
    
    def get_item_embedding(self, batch_item):
        all_users, all_items = self.computer()
        items = all_items[batch_item]
        return items

    def get_all_post_embedding(self):
        """get total embedding of users and items after convolution

        Returns
        -------
        users : 2-D FloatTensor (num. users x dim)
        items : 2-D FloatTensor (num. items x dim)
        """
        all_users, all_items = self.computer()
        return all_users, all_items

    def get_all_embedding(self):
        return self.get_all_post_embedding()
    
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
        
        u = all_users[batch_user]		# batch_size x dim
        i = all_items[batch_items]		# batch_size x k x dim
        
        score = torch.bmm(i, u.unsqueeze(-1)).squeeze(-1)   # batch_size, k
        
        return score
    
    def get_all_ratings(self):
        users, items = self.get_all_post_embedding()
        score_mat = torch.matmul(users, items.T)
        return score_mat

    def get_ratings(self, batch_user):
        users, items = self.get_all_post_embedding()
        users = users[batch_user]
        score_mat = torch.matmul(users, items.T)
        return score_mat


class BaseCTR(nn.Module):
    def __init__(self, args, feature_stastic):
        super().__init__()
        self.model_type = "ctr"
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.L2_weight = args.L2
        self.num_fields = len(feature_stastic) - 1
        self.embedding_layer = Embedding(self.embedding_dim, feature_stastic)
        self.embedding_layer_dim = self.embedding_dim * self.num_fields
    
    def forward_embed(self, sparse_input, dense_input=None):
        """Return the output of the embedding layer
        """
        dense_input = self.embedding_layer(sparse_input)
        return dense_input

    def forward(self, sparse_input, dense_input=None):
        dense_input = self.embedding_layer(sparse_input)
        logits = self.FeatureInteraction(dense_input, sparse_input)
        return logits
    
    def FeatureInteraction(self, dense_input, sparse_input):
        raise NotImplementedError

    def L2_Loss(self, weight):
        if weight == 0:
            return 0
        loss = 0
        for _, module in self.named_modules():
                for p_name, param in module.named_parameters():
                    if param.requires_grad:
                        if p_name in ["weight", "bias"]:
                            loss += torch.norm(param, p=2)
        return loss * weight
    
    def get_loss(self, logits, label):
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), label.squeeze(-1).float()) + self.L2_Loss(self.L2_weight)
        return loss
