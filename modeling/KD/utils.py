import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


def load_pkls(*fnames):
    success_flg = True
    pkls = []
    for fname in fnames:
        if os.path.exists(fname):
            pkls.append(pickle.load(open(fname, "rb")))
        else:
            pkls.append(None)
            success_flg = False
    return success_flg, *pkls


def dump_pkls(*pkl_fnames):
    for t in pkl_fnames:
        pkl, fname = t
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        pickle.dump(pkl, open(fname, "wb"))


def self_loop_graph(N):
    self_loop_idx = torch.stack([torch.arange(N), torch.arange(N)], dim=0)
    self_loop_data = torch.ones(self_loop_idx.size(1), dtype=torch.float)
    Graph = torch.sparse_coo_tensor(self_loop_idx, self_loop_data, (N, N), dtype=torch.float)
    return Graph.coalesce()


def pca(X:torch.tensor, n_components:int) -> torch.tensor:
    X = X.detach().cpu().numpy()
    pca = PCA(n_components=n_components)
    reduced_X = torch.from_numpy(pca.fit_transform(X)).cuda()
    return reduced_X


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(-1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
    

class Expert(nn.Module):
    def __init__(self, dims):
        super().__init__()
        mlps = []
        for i in range(len(dims) - 2):
            mlps.append(nn.Linear(dims[i], dims[i + 1]))
            mlps.append(nn.ReLU())
        mlps.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*mlps)

    def forward(self, x):
        return self.mlp(x)


class fullExpert(nn.Module):
    def __init__(self, dims, dropout_rate):
        super().__init__()
        mlps = []
        for i in range(len(dims) - 2):
            mlps.append(nn.Dropout(dropout_rate))
            mlps.append(nn.Linear(dims[i], dims[i + 1]))
            # mlps.append(nn.BatchNorm1d(dims[i + 1]))
            mlps.append(nn.ReLU())
        mlps.append(nn.Dropout(dropout_rate))
        mlps.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*mlps)

    def forward(self, x):
        return self.mlp(x)

class Projector(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, norm=True, dropout_rate=0., shallow=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.norm = norm
        if shallow:
            expert_dims = [self.input_dim, self.output_dim]
        else:
            expert_dims = [self.input_dim, (self.input_dim + self.output_dim) // 2, self.output_dim]
        self.experts = nn.ModuleList([fullExpert(expert_dims, dropout_rate) for i in range(self.num_experts)])

    def forward_experts(self, x, experts, reduction=True):
        expert_outputs = [experts[i](x).unsqueeze(-1) for i in range(self.num_experts)]
        expert_outputs = torch.cat(expert_outputs, -1)							# batch_size x output x num_experts
        if reduction:
            expert_outputs = expert_outputs.mean(-1)								# batch_size x output
            if self.norm:
                norm_s = expert_outputs.pow(2).sum(-1, keepdim=True).pow(1. / 2)        ## TODO: may be zero
                expert_outputs = expert_outputs.div(norm_s)                     # batch_size x output
        else:
            if self.norm:
                expert_outputs = expert_outputs.transpose(-1, -2)     # bs, num_experts, output
                norm_s = expert_outputs.pow(2).sum(-1, keepdim=True).pow(1. / 2)
                expert_outputs = expert_outputs.div(norm_s)                 # bs, num_experts, output
        return expert_outputs

    def forward(self, x, reduction=True):
        fea = self.forward_experts(x, self.experts, reduction)
        return fea


@torch.no_grad()
def CKA(X, Y):
    def HSCI(K, L):
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - 1. / n * torch.ones((n, n), device=K.device)
        KH = torch.mm(K, H)
        LH = torch.mm(L, H)
        hsci = 1. / ((n - 1) ** 2) * torch.trace(torch.mm(KH, LH))
        return hsci
    K = torch.mm(X, X.T)
    L = torch.mm(Y, Y.T)
    hsci = HSCI(K, L)
    varK = torch.sqrt(HSCI(K, K))
    varL = torch.sqrt(HSCI(L, L))
    return hsci / (varK * varL)


def info_abundance(X):
    lam = torch.linalg.svdvals(X)
    lam = torch.abs(lam)
    lam = lam / lam.max()
    return lam.sum(-1).mean().item()
