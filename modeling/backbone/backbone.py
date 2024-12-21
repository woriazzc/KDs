import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseRec, BaseGCN, BaseCTR
from .base_layer import BehaviorAggregator, MLP, LR, CrossNetComp, CINComp, AutoInt_AttentionLayer, EulerInteractionLayer, GateCrossLayer


"""
Recommendation Models
"""
class BPR(BaseRec):
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
        
        self.Graph = self.construct_graph()

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

class SCCF(BaseRec):
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
class SimpleX(BaseRec):
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
        
        self.Graph = self.construct_graph()

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



"""
CTR Prediction Models
"""
class FM(BaseCTR):
    def __init__(self, args, feature_stastic):
        super().__init__(args, feature_stastic)
        self.model_name = "fm"
        self.one_order = LR(feature_stastic)
    
    def FeatureInteraction(self, dense_input, sparse_input):
        fm = torch.sum(dense_input, dim=1) ** 2 - torch.sum(dense_input ** 2 , dim=1)
        logits = torch.sum(0.5 * fm, dim=1, keepdim=True) + self.one_order(sparse_input)
        return logits


class DeepFM(BaseCTR):
    def __init__(self, args, feature_stastic):
        super().__init__(args, feature_stastic)
        self.model_name = "deepfm"
        self.hidden_dims = args.hidden_dims
        self.dropout = args.dropout
        self.one_order = LR(feature_stastic)
        self.mlp = MLP(self.embedding_dim, feature_stastic, self.hidden_dims, self.dropout)
    
    def FeatureInteraction(self, dense_input, sparse_input):
        fm = torch.sum(dense_input, dim=1) ** 2 - torch.sum(dense_input ** 2, dim=1)
        logits = torch.sum(0.5 * fm, dim=1, keepdim=True) + self.one_order(sparse_input) + self.mlp(dense_input)
        return logits


class DNN(BaseCTR):
    def __init__(self, args, feature_stastic):
        super().__init__(args, feature_stastic)
        self.model_name = "dnn"
        self.hidden_dims = args.hidden_dims
        self.dropout = args.dropout
        self.mlp = MLP(self.embedding_dim, feature_stastic, self.hidden_dims, self.dropout)
    
    def FeatureInteraction(self, dense_input, sparse_input):
        logits = self.mlp(dense_input)
        return logits
    
    def get_layer_dim(self, layer):
        assert layer <= len(self.hidden_dims), "Layer id exceed maximum layers."
        if layer == 0: return self.embedding_layer_dim
        else: return self.hidden_dims[layer - 1]

    def forward_layer(self, sparse_input, layer):
        assert layer <= len(self.hidden_dims), "Layer id exceed maximum layers."
        feature = self.embedding_layer(sparse_input)
        feature = feature.reshape(feature.shape[0], -1)
        if layer == 0: return feature
        base = feature
        cross = feature
        for i in range(layer):
            cross = self.crossnet[i](base, cross)
        return cross
    
    @property
    def _penultimate_dim(self):
        return self.hidden_dims[-2]
    
    def forward_penultimate(self, sparse_input, dense_input=None):
        dense_input = self.embedding_layer(sparse_input)
        logits, feature = self.mlp(dense_input, penultimate=True)
        return feature

    def forward_all_feature(self, sparse_input, dense_input=None):
        dense_input = self.embedding_layer(sparse_input)
        all_features = [dense_input.reshape(dense_input.shape[0], -1)]
        all_features += self.mlp(dense_input, return_all=True)
        return all_features


class CrossNet(BaseCTR):
    def __init__(self, args, feature_stastic):
        super().__init__(args, feature_stastic)
        self.model_name = "crossnet"
        self.depth = args.depth
        self.crossnet = nn.ModuleList([CrossNetComp(self.embedding_dim, feature_stastic) for i in range(self.depth)])
        self.linear = nn.Linear((len(feature_stastic) - 1) * self.embedding_dim, 1)
        nn.init.normal_(self.linear.weight)
    
    def FeatureInteraction(self, feature, sparse_input):
        features = feature.reshape(feature.shape[0], -1)
        base = features
        cross = features
        for i in range(self.depth):
            cross = self.crossnet[i](base, cross)
        logits = self.linear(cross)
        return logits
    
    def get_layer_dim(self, layer):
        assert layer <= self.depth, "Layer id exceed maximum layers."
        if layer == 0: return self.embedding_layer_dim
        else: return self.linear.weight.shape[1]

    def forward_layer(self, sparse_input, layer):
        assert layer <= self.depth, "Layer id exceed maximum layers."
        feature = self.embedding_layer(sparse_input)
        feature = feature.reshape(feature.shape[0], -1)
        if layer == 0: return feature
        base = feature
        cross = feature
        for i in range(layer):
            cross = self.crossnet[i](base, cross)
        return cross
    
    @property
    def _penultimate_dim(self):
        # input dim of self.linear
        return self.linear.weight.shape[1]
    
    def forward_penultimate(self, sparse_input, dense_input=None):
        feature = self.embedding_layer(sparse_input)
        feature = feature.reshape(feature.shape[0], -1)
        base = feature
        cross = feature
        for i in range(self.depth):
            cross = self.crossnet[i](base, cross)
        return cross
    
    @property
    def _all_layer_dims(self):
        return [self.embedding_layer_dim] + [self.linear.weight.shape[1]] * self.depth

    def forward_all_feature(self, sparse_input, dense_input=None):
        feature = self.embedding_layer(sparse_input)
        feature = feature.reshape(feature.shape[0], -1)
        base = feature
        cross = feature
        all_features = [cross]
        for i in range(self.depth):
            cross = self.crossnet[i](base, cross)
            all_features.append(cross)
        return all_features


class DCNV2(BaseCTR):
    def __init__(self, args, feature_stastic):
        super().__init__(args, feature_stastic)
        self.model_name = "dcnv2"
        self.hidden_dims = args.hidden_dims
        self.dropout = args.dropout
        self.depth = args.depth
        self.mlp = MLP(self.embedding_dim, feature_stastic, self.hidden_dims, self.dropout)
        self.crossnet = nn.ModuleList([CrossNetComp(self.embedding_dim, feature_stastic) for i in range(self.depth)])
        self.linear = nn.Linear((len(feature_stastic) - 1) * self.embedding_dim, 1)
    
    @property
    def _penultimate_dim(self):
        return self.linear.weight.shape[1], self.hidden_dims[-2]
    
    def FeatureInteraction(self, feature, sparse_input):
        mlp = self.mlp(feature)
        feature = feature.reshape(feature.shape[0], -1)
        base = feature
        cross = feature
        for i in range(self.depth):
            cross = self.crossnet[i](base, cross)
        logits = self.linear(cross) + mlp
        return logits
    
    def forward_penultimate(self, sparse_input, dense_input=None):
        dense_input = self.embedding_layer(sparse_input)
        logits, feature = self.mlp(dense_input, penultimate=True)
        dense_input = dense_input.reshape(dense_input.shape[0], -1)
        base = dense_input
        cross = dense_input
        for i in range(self.depth):
            cross = self.crossnet[i](base, cross)
        return cross, feature


class DAGFM(BaseCTR):
    def __init__(self, args, feature_stastic):
        super().__init__(args, feature_stastic)
        self.model_name = "dagfm"
        self.type = args.type
        self.depth = args.depth
        field_num = len(feature_stastic) - 1
        if type == 'inner':
            self.p = nn.ParameterList([nn.Parameter(torch.randn(field_num, field_num, self.embedding_dim)) for _ in range(self.depth)])
        elif type == 'outer':
            self.p = nn.ParameterList([nn.Parameter(torch.randn(field_num, field_num, self.embedding_dim)) for _ in range(self.depth)])
            self.q = nn.ParameterList([nn.Parameter(torch.randn(field_num, field_num, self.embedding_dim)) for _ in range(self.depth)])
            for _ in range(self.depth):
                nn.init.xavier_normal_(self.p[_], gain=1.414)
                nn.init.xavier_normal_(self.q[_], gain=1.414)
        self.adj_matrix = torch.zeros(field_num, field_num, self.embedding_dim).cuda()
        for i in range(field_num):
            for j in range(i, field_num):
                self.adj_matrix[i, j, :] += 1
        self.connect_layer = nn.Parameter(torch.eye(field_num).float())
        self.linear = nn.Linear(field_num * (self.depth + 1), 1)
    
    def FeatureInteraction(self, feature, sparse_input):
        init_state = self.connect_layer @ feature
        h0, ht = init_state, init_state
        state = [torch.sum(init_state, dim=-1)]
        for i in range(self.depth):
            if self.type == 'inner':
                aggr = torch.einsum('bfd,fsd->bsd', ht, self.p[i] * self.adj_matrix)
                ht = h0 * aggr
            elif self.type == 'outer':
                term = torch.einsum('bfd,fsd->bfs', ht, self.p[i] * self.adj_matrix)
                aggr = torch.einsum('bfs,fsd->bsd', term, self.q[i])
                ht = h0 * aggr
            state.append(torch.sum(ht, -1))
            
        state = torch.cat(state, dim=-1)
        logits = self.linear(state)
        return logits


class CIN(BaseCTR):
    def __init__(self, args, feature_stastic):
        super().__init__(args, feature_stastic)
        self.model_name = "cin"
        self.cin_dims = args.cin_dims
        self.cinlist = [len(feature_stastic) - 1] + self.cin_dims
        self.cin = nn.ModuleList([CINComp(self.cinlist[i], self.cinlist[i + 1], feature_stastic) for i in range(0, len(self.cinlist) - 1)])
        self.linear = nn.Linear(sum(self.cinlist) - self.cinlist[0], 1)
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
    
    def FeatureInteraction(self, feature, sparse_input):
        base = feature
        x = feature
        p = []
        for comp in self.cin:
            x = comp(x, base)
            p.append(torch.sum(x, dim=-1))
        p = torch.cat(p, dim=-1)
        logits = self.linear(p)
        return logits
    
    @property
    def _penultimate_dim(self):
        return self.linear.weight.shape[1]

    def forward_penultimate(self, sparse_input, dense_input=None):
        feature = self.embedding_layer(sparse_input)
        base = feature
        x = feature
        p = []
        for comp in self.cin:
            x = comp(x, base)
            p.append(torch.sum(x, dim=-1))
        p = torch.cat(p, dim=-1)
        return p
    
    def forward_all_feature(self, sparse_input, dense_input=None):
        feature = self.embedding_layer(sparse_input)
        base = feature
        all_features = [feature.reshape(feature.shape[0], -1)]
        x = feature
        for comp in self.cin:
            x = comp(x, base)
            all_features.append(torch.sum(x, dim=-1))
        return all_features


class XDeepFM(BaseCTR):
    def __init__(self, args, feature_stastic):
        super().__init__(args, feature_stastic)
        self.model_name = "xdeepfm"
        self.hidden_dims = args.hidden_dims
        self.dropout = args.dropout
        self.cin_dims = args.cin_dims
        self.mlp = MLP(self.embedding_dim, feature_stastic, self.hidden_dims, self.dropout)
        self.one_order = LR(feature_stastic)
        self.cinlist = [len(feature_stastic) - 1] + self.cin_dims
        self.cin = nn.ModuleList([CINComp(self.cinlist[i], self.cinlist[i + 1], feature_stastic) for i in range(0, len(self.cinlist) - 1)])
        self.linear = nn.Linear(sum(self.cinlist) - self.cinlist[0], 1)
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)

    def FeatureInteraction(self, feature, sparse_input):
        mlp = self.mlp(feature)
        base = feature
        x = feature
        p = []
        for comp in self.cin:
            x = comp(x, base)
            p.append(torch.sum(x, dim=-1))
        p = torch.cat(p, dim=-1)
        cin = self.linear(p)
        logits = cin + mlp + self.one_order(sparse_input)
        return logits


class CrossCIN(BaseCTR):
    def __init__(self, args, feature_stastic):
        super().__init__(args, feature_stastic)
        self.model_name = "crosscin"
        self.crossnet = CrossNet(args, feature_stastic)
        self.cin = CIN(args, feature_stastic)
    
    def FeatureInteraction(self, feature, sparse_input):
        logits_crossnet = self.crossnet.FeatureInteraction(feature, sparse_input)
        logits_cin = self.cin.FeatureInteraction(feature, sparse_input)
        return logits_crossnet + logits_cin


class AutoInt(BaseCTR):
    def __init__(self, args, feature_stastic):
        super().__init__(args, feature_stastic)
        self.model_name = "autoint"
        self.headNum = args.num_head
        self.LayerNum = args.depth
        self.att_emb = args.attention_dim
        self.hidden_dims = args.hidden_dims
        self.dropout = args.dropout
        self.featureNum = len(feature_stastic) - 1
        self.input_emb = self.att_emb * self.headNum
        self.interacting = nn.Sequential(*[AutoInt_AttentionLayer(self.headNum, self.att_emb, self.input_emb if _ != 0 else self.embedding_dim) for _ in range(self.LayerNum)])
        self.mlp = MLP(self.embedding_dim, feature_stastic, self.hidden_dims, self.dropout)
        self.linear = nn.Linear(self.input_emb * self.featureNum, 1)
        nn.init.xavier_uniform_(self.linear.weight, gain=1.414)

    def FeatureInteraction(self, feature, sparse_input):
        mlp = self.mlp(feature)        
        attention = self.interacting(feature) #[b,f,h*d]
        attention = attention.reshape(feature.shape[0], -1)
        logits = self.linear(attention) + mlp
        return logits


class EulerNet(BaseCTR):
    def __init__(self, args, feature_stastic):
        super().__init__(args, feature_stastic)
        self.model_name = "eulernet"
        self.order_list = args.order_list
        self.apply_norm = args.eulernet_norm
        self.drop_ex = args.eulernet_dropex
        self.drop_im = args.eulernet_dropim
        field_num = len(feature_stastic) - 1
        shape_list = [self.embedding_dim * field_num] + [num_neurons * self.embedding_dim for num_neurons in self.order_list]

        interaction_shapes = []
        for inshape, outshape in zip(shape_list[:-1], shape_list[1:]):
            interaction_shapes.append(EulerInteractionLayer(inshape, outshape, self.embedding_dim, self.apply_norm, self.drop_ex, self.drop_im))

        self.Euler_interaction_layers = nn.Sequential(*interaction_shapes)
        self.mu = nn.Parameter(torch.ones(1, field_num, 1))
        self.reg = nn.Linear(shape_list[-1], 1)
        nn.init.xavier_normal_(self.reg.weight)

    def FeatureInteraction(self, feature, sparse_input):
        r, p = self.mu * torch.cos(feature), self.mu * torch.sin(feature)
        o_r, o_p = self.Euler_interaction_layers((r, p))
        o_r, o_p = o_r.reshape(o_r.shape[0], -1), o_p.reshape(o_p.shape[0], -1)
        re, im = self.reg(o_r), self.reg(o_p)
        logits = re + im
        return logits


class GateCrossNet(BaseCTR):
    def __init__(self, args, feature_stastic):
        super().__init__(args, feature_stastic)
        self.model_name = "gatecrossnet"
        self.depth = args.depth
        self.crossnet = nn.ModuleList([GateCrossLayer(self.embedding_dim, feature_stastic) for i in range(self.depth)])
        self.linear = nn.Linear((len(feature_stastic) - 1) * self.embedding_dim, 1)
        nn.init.normal_(self.linear.weight)
    
    def FeatureInteraction(self, feature, sparse_input):
        features = feature.reshape(feature.shape[0], -1)
        base = features
        cross = features
        for i in range(self.depth):
            cross = self.crossnet[i](base, cross)
        logits = self.linear(cross)
        return logits
    
    def get_layer_dim(self, layer):
        assert layer <= self.depth, "Layer id exceed maximum layers."
        if layer == 0: return self.embedding_layer_dim
        else: return self.linear.weight.shape[1]

    def forward_layer(self, sparse_input, layer):
        assert layer <= self.depth, "Layer id exceed maximum layers."
        feature = self.embedding_layer(sparse_input)
        feature = feature.reshape(feature.shape[0], -1)
        if layer == 0: return feature
        base = feature
        cross = feature
        for i in range(layer):
            cross = self.crossnet[i](base, cross)
        return cross
    
    @property
    def _penultimate_dim(self):
        # input dim of self.linear
        return self.linear.weight.shape[1]
    
    def forward_penultimate(self, sparse_input, dense_input=None):
        feature = self.embedding_layer(sparse_input)
        feature = feature.reshape(feature.shape[0], -1)
        base = feature
        cross = feature
        for i in range(self.depth):
            cross = self.crossnet[i](base, cross)
        return cross
    
    @property
    def _all_layer_dims(self):
        return [self.embedding_layer_dim] + [self.linear.weight.shape[1]] * self.depth

    def forward_all_feature(self, sparse_input, dense_input=None):
        feature = self.embedding_layer(sparse_input)
        feature = feature.reshape(feature.shape[0], -1)
        base = feature
        cross = feature
        all_features = [cross]
        for i in range(self.depth):
            cross = self.crossnet[i](base, cross)
            all_features.append(cross)
        return all_features
