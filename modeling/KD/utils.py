import os
import pickle
import mlflow
import numpy as np
import scipy.linalg
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import  RBF 

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def cal_rsd(X, Y):
        Us, Ss, Vs = torch.svd(X.T)
        Ut, St, Vt = torch.svd(Y.T)
        Ps, cospa, Pt = torch.svd(torch.mm(Us.T, Ut))
        cospa = torch.round(torch.minimum(cospa, torch.tensor(1.)), decimals=4)
        sinpa = torch.sqrt(1. - torch.pow(cospa, 2))
        rsd = torch.norm(sinpa, 1)
        # bmp = torch.norm(Ps.abs() - Pt.abs(), 2)
        dis = rsd
        return dis


def log_rsd(func):
    def wrapper(self, data, label):
        loss = func(self, data, label)
        if self.args.verbose:
            with torch.no_grad():
                S_emb = self.student.forward_penultimate(data)
                T_emb = self.teacher.forward_penultimate(data)
                rsd = cal_rsd(S_emb, T_emb)
                mlflow.log_metric("rsd", rsd.item())
                mlflow.log_metric("loss", loss.item())
        return loss
    return wrapper


def log_norm(func):
    def wrapper(self, data, label):
        loss = func(self, data, label)
        if self.args.verbose:
            with torch.no_grad():
                S_embs = self.student.forward_all_feature(data)
                T_embs = self.teacher.forward_all_feature(data)
                for i, S_emb in enumerate(S_embs):
                    S_norm = torch.norm(S_emb, p=2, dim=-1).mean()
                    mlflow.log_metric(f"S_norm_{i}", S_norm.item())
                for i, T_emb in enumerate(T_embs):
                    T_norm = torch.norm(T_emb, p=2, dim=-1).mean()
                    mlflow.log_metric(f"T_norm_{i}", T_norm.item())
        return loss
    return wrapper


def save_embedding(func):
    def wrapper(self, data, label):
        loss = func(self, data, label)
        if self.args.verbose:
            with torch.no_grad():
                S_embs = self.student.embedding_layer.embedding
                T_embs = self.teacher.embedding_layer.embedding
                save_dir = os.path.join(self.args.CRAFT_DIR, self.args.dataset, self.teacher.model_name, self.student.model_name, self.model_name)
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                if not hasattr(self, "save_embed_step"):
                    self.save_embed_step = 0
                    torch.save(T_embs, os.path.join(save_dir, "T_embs.pt"))
                if self.save_embed_step % 1000 == 0:
                    torch.save(S_embs, os.path.join(save_dir, f"S_embs_ep{self.save_embed_step}.pt"))
                self.save_embed_step += 1
        return loss
    return wrapper


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
    
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss


def log_mmd(func):
    def wrapper(self, data, label):
        loss = func(self, data, label)
        if self.args.verbose:
            with torch.no_grad():
                if not hasattr(self, "log_mmd_step"):
                    self.log_mmd_step = 0
                log_interval = 2
                if self.log_mmd_step % log_interval == 0:
                    S_emb = self.student.forward_penultimate(data)
                    T_emb = self.teacher.forward_penultimate(data)
                    assert S_emb.shape[0] == label.shape[0]
                    S_emb, T_emb = S_emb[:512], T_emb[:512]
                    mmd = MMD_loss()(S_emb, T_emb)
                    mlflow.log_metric("mmd", mmd.item(), step=self.log_mmd_step // log_interval)
                self.log_mmd_step += 1
        return loss
    return wrapper


# https://github.com/MaterialsInformaticsDemo/TCA/blob/main/Python/TCA.py
class TCA():
    def __init__(self, dim=30, lamda=1, gamma=1):
        '''
        :param dim: data dimension after projection
        :param lamb: lambda value, Lagrange multiplier
        :param gamma: length scale for rbf kernel
        '''
        self.dim = dim
        self.lamda = lamda
        self.kernel = 0.5 * RBF(gamma, "fixed")

    def fit(self, Xs, Xt):
        '''
        :param Xs: ns * m_feature, source domain data 
        :param Xt: nt * m_feature, target domain data
        Projecting Xs and Xt to a lower dimension by TCA
        source/target domain data expressed in a mapping space
        :return: Xs_new and Xt_new 
        '''
        # formular in paper Domain Adaptation via Transfer Component Analysis
        # Eq.(2)
        X = np.vstack((Xs, Xt))
        K = self.kernel(X)
        # cal matrix L
        ns, nt = len(Xs), len(Xt)
        if self.dim > (ns + nt):
            raise ValueError('The maximum number of dimensions should be smaller than', (ns + nt))
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        L = e * e.T
        # cal centering matrix H page 202 the last pargraph at left side
        n, _ = X.shape
        H = np.eye(n) - 1 / n * np.ones((n, n))
        # page 202 the last pargraph at right side
        matrix = np.linalg.inv(K @ L @ K + self.lamda * np.eye(n)) @ K @ H @ K
        # cal eigenvalues : w, eigenvectors : V
        w, V = scipy.linalg.eig(matrix)
        w, V = w.real, V.real
        # peak out the first self.dim components
        ind = np.argsort(abs(w))[::-1]
        A = V[:, ind[:self.dim]]
        # output the mapped data
        Z = K @ A
        Xs_new, Xt_new = Z[:ns, :], Z[ns:, :]
        return Xs_new, Xt_new
