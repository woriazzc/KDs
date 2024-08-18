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


# https://github.com/frnetnetwork/frnet/blob/main/model/BasiclLayer.py
class FeaturesLinear(torch.nn.Module):
    """
    Linear regression layer for CTR prediction.
    """
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)

    def forward(self, x):
        """
        :param x: B,F
        :return: B,E
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FactorizationMachine(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: B,F,E
        """
        # 因为x都是1，所以不需要乘以x: 和的平方 - 平方的和
        square_of_sum = torch.sum(x, dim=1) ** 2  # B，embed_dim
        sum_of_square = torch.sum(x ** 2, dim=1)  # B，embed_dim
        # square of sum - sum of square
        ix = square_of_sum - sum_of_square  # B,embed_dim
        if self.reduce_sum:
            # For NFM, reduce_sum = False
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


# Embedding Layer
class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        """
        :param field_dims: list
        :param embed_dim
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        self._init_weight_()

    def _init_weight_(self):
        """ weights initialization"""
        nn.init.normal_(self.embedding.weight, std=0.01)
        # nn.init.xavier_normal_nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, x):
        """
        :param x: B,F
        :return: B,F,E
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout=0.5, output_layer=False):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim

        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        # 使用 *，
        self.mlp = torch.nn.Sequential(*layers)
        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size,num_fields*embed_dim)``
        """
        return self.mlp(x)


class InnerProductNetwork(torch.nn.Module):

    def __init__(self,num_fields,is_sum=True):
        super(InnerProductNetwork, self).__init__()
        self.is_sum = is_sum
        self.num_fields = num_fields
        self.row, self.col = list(), list()

        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                self.row.append(i), self.col.append(j)

    def forward(self, x):
        if self.is_sum == True:
            # 默认求和最原始的方式
            return torch.sum(x[:, self.row] * x[:, self.col], dim=2)  # B,1/2* nf*(nf-1)
        else:
            #  以下： 如果不求和 B,1/2* nf*(nf-1), K
            return x[:, self.row] * x[:, self.col]


class OuterProductNetwork(torch.nn.Module):
    def __init__(self, num_fields, embed_dim, kernel_type='num'):
        super().__init__()

        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type

        self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))

        num_field = num_fields
        self.row, self.col = list(), list()
        for i in range(num_field - 1):
            for j in range(i + 1, num_field):
                self.row.append(i), self.col.append(j)
        torch.nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        p, q = x[:, self.row], x[:, self.col]  # B,n,emb

        if self.kernel_type == 'mat':
            #  p [b,1,num_ix,e]
            #  kernel [e, num_ix, e]
            kp = torch.sum(p.unsqueeze(1) * self.kernel,dim=-1).permute(0,2,1)  #b,num_ix,e
            # #b,num_ix,e
            return torch.sum(kp * q, -1)
        else:
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)


class Embedding(nn.Module):
    def __init__(self, embedding_dim, feature_stastic):
        super().__init__()
        self.embedding: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()
        for feature , numb in feature_stastic.items():
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
