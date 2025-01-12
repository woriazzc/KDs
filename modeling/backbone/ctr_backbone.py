import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseCTR
from .base_layer import (MLP, LR, CrossNetComp, CINComp, AutoInt_AttentionLayer, EulerInteractionLayer, GateCrossLayer)


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
