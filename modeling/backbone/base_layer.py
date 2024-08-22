import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# For SimpleX, from https://github.com/reczoo/RecZoo/blob/main/matching/cf/SimpleX/src/SimpleX.py
class BehaviorAggregator(nn.Module):
    def __init__(self, embedding_dim, gamma=0.5, aggregator="mean", dropout_rate=0.):
        super(BehaviorAggregator, self).__init__()
        self.aggregator = aggregator
        self.gamma = gamma
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        if self.aggregator in ["user_attention", "self_attention"]:
            self.W_k = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                     nn.Tanh())
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
            if self.aggregator == "self_attention":
                self.W_q = nn.Parameter(torch.Tensor(embedding_dim, 1))
                nn.init.xavier_normal_(self.W_q)

    def forward(self, uid_emb, sequence_emb):
        out = uid_emb
        if self.aggregator == "mean":
            out = self.average_pooling(sequence_emb)
        elif self.aggregator == "user_attention":
            out = self.user_attention(uid_emb, sequence_emb)
        elif self.aggregator == "self_attention":
            out = self.self_attention(sequence_emb)
        return self.gamma * uid_emb + (1 - self.gamma) * out

    def user_attention(self, uid_emb, sequence_emb):
        key = self.W_k(sequence_emb) # b x seq_len x attention_dim
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.bmm(key, uid_emb.unsqueeze(-1)).squeeze(-1) # b x seq_len
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_v(output)

    def self_attention(self, sequence_emb):
        key = self.W_k(sequence_emb) # b x seq_len x attention_dim
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.matmul(key, self.W_q).squeeze(-1) # b x seq_len
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_v(output)

    def average_pooling(self, sequence_emb):
        mask = sequence_emb.sum(dim=-1) != 0
        mean = sequence_emb.sum(dim=1) / (mask.float().sum(dim=-1, keepdim=True) + 1.e-9)
        return self.W_v(mean)

    def masked_softmax(self, X, mask):
        # use the following softmax to avoid nans when a sequence is entirely masked
        X = X.masked_fill_(mask, 0)
        e_X = torch.exp(X)
        return e_X / (e_X.sum(dim=1, keepdim=True) + 1.e-9)


class Embedding(nn.Module):
    def __init__(self, embedding_dim, feature_stastic):
        super().__init__()
        self.embedding: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()
        for feature, numb in feature_stastic.items():
            if feature != 'label':
                self.embedding[feature] = nn.Embedding(numb + 1, embedding_dim)
        for _, value in self.embedding.items():
            nn.init.xavier_uniform_(value.weight)

    def forward(self, data):
        out = []
        for name, raw in data.items():
            if name != 'label':
                    out.append(self.embedding[name](raw.long().cuda())[:, None, :])
        return torch.cat(out, dim=-2)


class MLP(nn.Module):
    def __init__(self, embedding_dim, feature_stastic, hidden_dims, dropout, act=None):
        super().__init__()
        layers = []
        Shape = [(len(feature_stastic) - 1) * embedding_dim] + hidden_dims
        for i in range(0, len(Shape) - 2):
            hidden = nn.Linear(Shape[i], Shape[i + 1], bias= True)
            nn.init.normal_(hidden.weight, mean=0, std=0.01)
            layers.append(hidden)
            layers.append(nn.Dropout(p=dropout))
            layers.append(act if act is not None else nn.ReLU(inplace=False))
        self.Final = nn.Linear(Shape[-2], Shape[-1], bias=True)
        nn.init.xavier_normal_(self.Final.weight, gain=1.414)
        # layers.append(Final)
        self.net = nn.Sequential(*layers)
    
    def forward(self, x : torch.Tensor, penultimate=False):
        x = x.reshape(x.shape[0], -1)
        feature = self.net(x)
        ret = self.Final(feature)
        if penultimate:
            return ret, feature
        else:
            return ret
    

class LR(nn.Module):
    def __init__(self, feature_stastic):
        super(LR,self).__init__()
        self.embedding: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()
        for feature, numb in feature_stastic.items():
            if feature != 'label':
                self.embedding[feature] = torch.nn.Embedding(numb + 1, 1)
        
        for _, value in self.embedding.items():
            nn.init.xavier_normal_(value.weight)
    
    def forward(self, data):
        out = []
        for name, raw in data.items():
            if name != 'label':
                out.append(self.embedding[name](raw.long().cuda())[:, None, :])
        out = torch.cat(out, dim=-2)
        return torch.sum(out, dim=1)


class CrossNetComp(nn.Module):
    def __init__(self, embedding_dim, feature_stastic):
        super(CrossNetComp, self).__init__()
        hiddenSize = (len(feature_stastic) - 1) * embedding_dim
        self.W = nn.Linear(hiddenSize, hiddenSize)
        nn.init.xavier_normal_(self.W.weight, gain=1.414)
        
    def forward(self, base, cross):
        result = base * self.W(cross) + cross
        return result


class CINComp(nn.Module):
    def __init__(self, indim, outdim, feature_stastic):
        super(CINComp, self).__init__()
        basedim = len(feature_stastic) - 1
        self.conv = nn.Conv1d(indim * basedim, outdim, 1)
    
    def forward(self, infeature, base):
        return self.conv(
            (infeature[:, :, None, :] * base[:, None, :, :]) \
            .reshape(infeature.shape[0], infeature.shape[1] * base.shape[1], -1)
        )
