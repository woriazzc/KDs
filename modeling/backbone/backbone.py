import os
import math
import pickle
import random
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from fuxictr.pytorch.models import BaseModel as fuxiBaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, FactorizationMachine, CrossNetV2, CrossNetMix

from .base_model import BaseRec, BaseGCN, BaseCTR
from .utils import convert_sp_mat_to_sp_tensor, load_pkls, dump_pkls
from .base_layer import BehaviorAggregator, MultiLayerPerceptron


"""
Recommendation Models
"""
class BPR(BaseRec):
    def __init__(self, dataset, args):
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
        
        self.embedding_dim = args.embedding_dim
        self.num_layers = args.num_layers
        self.keep_prob = getattr(args, "keep_prob", 0.)
        self.A_split = getattr(args, "A_split", False)
        self.dropout = getattr(args, "dropout", False)
        self.init_std = args.init_std

        self.user_emb = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.item_emb = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
        
        self.Graph = self.construct_graph()

        self.reset_para()
    
    def reset_para(self):
        nn.init.normal_(self.user_emb.weight, std=self.init_std)
        nn.init.normal_(self.item_emb.weight, std=self.init_std)

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
    
    def get_all_pre_embedding(self):
        users = self.user_emb(self.user_list)
        items = self.item_emb(self.item_list)
        
        return users, items


# Refer to https://github.com/reczoo/RecZoo/blob/main/matching/cf/SimpleX/src/SimpleX.py
class SimpleX(BaseRec):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
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



"""
CTR Prediction Models
"""
# https://github.com/frnetnetwork/frnet/blob/main/model/DeepFM.py
class DeepFM(BaseCTR):
    def __init__(self, args):
        super().__init__(args)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embed_output_size = len(self.field_dims) * self.embed_dim
        self.dropout = args.dropout
        self.mlp_layers = args.mlp_layers
        self.mlp = MultiLayerPerceptron(self.embed_output_size, self.mlp_layers, self.dropout, output_layer=True)

    def forward(self, x):
        """
        :param x: B,F
        :return:
        """
        x_embed = self.embedding(x)  # B,F,E
        x_out = self.lr(x) + self.fm(x_embed) + self.mlp(x_embed.view(x.size(0), -1))
        return x_out.squeeze(-1)


def sum_emb_out_dim(feature_map, emb_dim, feature_source=[]):
        if type(feature_source) != list:
            feature_source = [feature_source]
        total_dim = 0
        for feature, feature_spec in feature_map.features.items():
            if feature_spec["type"] == "meta":
                continue
            if len(feature_source) == 0 or feature_spec.get("source") in feature_source:
                total_dim += feature_spec.get("emb_output_dim",
                                              feature_spec.get("embedding_dim", 
                                                               emb_dim))
        return total_dim


class DeepFM(fuxiBaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DeepFM", 
                 gpu=0, 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        kwargs.update({"verbose": 0, "task": "binary_classification", "model_root": kwargs["CKPT_DIR"], "metrics": ["logloss", "AUC"]})
        super(DeepFM, self).__init__(feature_map, 
                                     model_id=model_id, 
                                     gpu=gpu, 
                                     embedding_regularizer=embedding_regularizer, 
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.fm = FactorizationMachine(feature_map)
        self.mlp = MLP_Block(input_dim=sum_emb_out_dim(feature_map, embedding_dim),
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        logit = self.fm(X, feature_emb)
        logit += self.mlp(feature_emb.flatten(start_dim=1))
        return logit
    
    def get_loss(self, output, labels):
        """Compute the loss function with the model output

        Parameters
        ----------
        output : 
            model output (results of forward function)
        labels:
            label of this sample

        Returns
        -------
        loss : float
        """
        loss = F.binary_cross_entropy_with_logits(output, labels.float())
        loss += self.regularization_loss()
        return loss


class DCNv2(fuxiBaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DCNv2", 
                 gpu=0,
                 model_structure="parallel",
                 use_low_rank_mixture=False,
                 low_rank=32,
                 num_experts=4,
                 embedding_dim=10, 
                 stacked_dnn_hidden_units=[], 
                 parallel_dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None,
                 net_regularizer=None, 
                 **kwargs):
        kwargs.update({"verbose": 0, "task": "binary_classification", "model_root": kwargs["CKPT_DIR"], "metrics": ["logloss", "AUC"]})
        super(DCNv2, self).__init__(feature_map, 
                                    model_id=model_id, 
                                    gpu=gpu, 
                                    embedding_regularizer=embedding_regularizer, 
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = sum_emb_out_dim(feature_map, embedding_dim)
        if use_low_rank_mixture:
            self.crossnet = CrossNetMix(input_dim, num_cross_layers, low_rank=low_rank, num_experts=num_experts)
        else:
            self.crossnet = CrossNetV2(input_dim, num_cross_layers)
        self.model_structure = model_structure
        assert self.model_structure in ["crossnet_only", "stacked", "parallel", "stacked_parallel"], \
               "model_structure={} not supported!".format(self.model_structure)
        if self.model_structure in ["stacked", "stacked_parallel"]:
            self.stacked_dnn = MLP_Block(input_dim=input_dim,
                                         output_dim=None, # output hidden layer
                                         hidden_units=stacked_dnn_hidden_units,
                                         hidden_activations=dnn_activations,
                                         output_activation=None, 
                                         dropout_rates=net_dropout,
                                         batch_norm=batch_norm)
            final_dim = stacked_dnn_hidden_units[-1]
        if self.model_structure in ["parallel", "stacked_parallel"]:
            self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                          output_dim=None, # output hidden layer
                                          hidden_units=parallel_dnn_hidden_units,
                                          hidden_activations=dnn_activations,
                                          output_activation=None, 
                                          dropout_rates=net_dropout, 
                                          batch_norm=batch_norm)
            final_dim = input_dim + parallel_dnn_hidden_units[-1]
        if self.model_structure == "stacked_parallel":
            final_dim = stacked_dnn_hidden_units[-1] + parallel_dnn_hidden_units[-1]
        if self.model_structure == "crossnet_only": # only CrossNet
            final_dim = input_dim
        self.fc = nn.Linear(final_dim, 1)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=True)
        cross_out = self.crossnet(feature_emb)
        if self.model_structure == "crossnet_only":
            final_out = cross_out
        elif self.model_structure == "stacked":
            final_out = self.stacked_dnn(cross_out)
        elif self.model_structure == "parallel":
            dnn_out = self.parallel_dnn(feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        elif self.model_structure == "stacked_parallel":
            final_out = torch.cat([self.stacked_dnn(cross_out), self.parallel_dnn(feature_emb)], dim=-1)
        logit = self.fc(final_out)
        return logit
    
    def get_loss(self, output, labels):
        loss = F.binary_cross_entropy_with_logits(output, labels.float())
        loss += self.regularization_loss()
        return loss


class DNN(fuxiBaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DNN", 
                 gpu=0, 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        kwargs.update({"verbose": 0, "task": "binary_classification", "model_root": kwargs["CKPT_DIR"], "metrics": ["logloss", "AUC"]})
        super(DNN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.mlp = MLP_Block(input_dim=sum_emb_out_dim(feature_map, embedding_dim),
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=True)
        logit = self.mlp(feature_emb)
        return logit

    def get_loss(self, output, labels):
        loss = F.binary_cross_entropy_with_logits(output, labels.float())
        loss += self.regularization_loss()
        return loss
    