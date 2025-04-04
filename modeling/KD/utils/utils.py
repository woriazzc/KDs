import os
import yaml
import pickle
import random
import mlflow
import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse import csr_matrix
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


def self_loop_graph(N, values=None):
    if values is None:
        values = torch.ones(N, dtype=torch.float)
    assert len(values) == N
    values = values.cpu()
    self_loop_idx = torch.stack([torch.arange(N), torch.arange(N)], dim=0)
    Graph = torch.sparse_coo_tensor(self_loop_idx, values, (N, N), dtype=torch.float)
    return Graph.coalesce()


def graph_mm(graph, X):
        return torch.sparse.mm(graph, X)
        # return torch.mm(graph.to_dense(), X)


def pca(X:torch.tensor, n_components:int) -> torch.tensor:
    device = X.device
    X = X.detach().cpu().numpy()
    pca = PCA(n_components=n_components)
    reduced_X = torch.from_numpy(pca.fit_transform(X)).to(device)
    return reduced_X


def frequency(Adj:torch.sparse_coo_tensor, X:torch.Tensor, k:int=200):
    """ X: shape (|V|, d)
    """
    # U, S, V = torch.svd_lowrank(Adj, q=k, niter=30)
    S_low, U_low = torch.lobpcg(Adj, k=k, largest=True)
    S_high, U_high = torch.lobpcg(Adj, k=k, largest=False)
    S = torch.cat([S_low, S_high])
    U = torch.cat([U_low, U_high], dim=-1)  # (|V|, 2k)
    freq = X.T.mm(U)  # (d, 2k)
    return freq


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
            if dropout_rate > 0:
                mlps.append(nn.Dropout(dropout_rate))
            mlps.append(nn.Linear(dims[i], dims[i + 1]))
            # mlps.append(nn.BatchNorm1d(dims[i + 1]))
            mlps.append(nn.ReLU())
        if dropout_rate > 0:
            mlps.append(nn.Dropout(dropout_rate))
        mlps.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*mlps)

    def forward(self, x):
        return self.mlp(x)

class Projector(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, norm=False, dropout_rate=0., shallow=False, hidden_dim_ratio=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.norm = norm
        self.hidden_dim_ratio = hidden_dim_ratio
        if shallow:
            expert_dims = [self.input_dim, self.output_dim]
        else:
            expert_dims = [self.input_dim, int(output_dim * self.hidden_dim_ratio), self.output_dim]
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
    return lam.sum().item()


def cal_rsd(X, Y, ratio=0.9):
    Us, Ss, Vs = torch.svd(X.T)
    Ut, St, Vt = torch.svd(Y.T)
    total_Ss, total_St = torch.sum(Ss), torch.sum(St)
    cumu_Ss, cumu_St = torch.cumsum(Ss, dim=0), torch.cumsum(St, dim=0)
    thresh_s, thresh_t = ratio * total_Ss, ratio * total_St
    K_s, K_t = torch.where(cumu_Ss >= thresh_s)[0][0] + 1, torch.where(cumu_St >= thresh_t)[0][0] + 1
    Us, Ut = Us[:, :K_s], Ut[:, :K_t]
    # mlflow.log_metrics({"Ss":Ss[:K_s].sum().item() / Ss.sum().item(), "St":St[:K_t].sum().item() / St.sum().item()})
    # mlflow.log_metrics({"Ks":K_s.item(), "Kt":K_t.item()})
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
                rsd = cal_rsd(S_emb, T_emb, 0.5)
                mlflow.log_metric("rsd", rsd.item())
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
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.pow(2).sum()
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
        else:
            raise NotImplementedError("Unexpected kernel_type.")


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


# https://github.com/jindongwang/transferlearning/blob/master/code/traditional/JDA/JDA.py
class TCA:
    def __init__(self, dim=20, lmbda=1, K=512, norm=True):
        '''
        Init func
        :param dim: dimension after transfer
        :param lmbda: lambda value in equation
        :param K: number of samples for trunctation
        '''
        self.dim = dim
        self.lmbda = lmbda
        self.K = K
        self.norm = norm

    def fit_transform(self, X_s, X_t, label_o):
        '''
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new, Xt_new
        '''
        Xs = X_s[:self.K].detach().cpu().numpy()
        Xt = X_t[:self.K].detach().cpu().numpy()
        label = label_o[:self.K].detach().cpu().numpy()
        X = np.hstack((Xs.T, Xt.T))
        if self.norm:
            X /= np.linalg.norm(X, axis=0)    # Layer normalize
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        H = np.eye(n) - 1 / n * np.ones((n, n))
        M = e * e.T * 2
        for c in [0, 1]:
            e = np.zeros((n, 1))
            tt = (label == c)
            ind = np.where(tt == True)
            e[ind] = 1 / len(label[ind])
            inds = [item + ns for item in ind]
            e[tuple(inds)] = -1 / len(label[ind])
            e[np.isinf(e)] = 0
            M = M + np.dot(e, e.T)
        M = M / np.linalg.norm(M, 'fro')
        n_eye = m
        a, b = np.linalg.multi_dot([X, M, X.T]) + self.lmbda * np.eye(n_eye), np.linalg.multi_dot([X, H, X.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]    # n_feature, dim
        A = torch.from_numpy(A).to(X_s.device).to(X_s.dtype)
        if self.norm:
            X_s_new = (X_s / torch.linalg.norm(X_s, dim=-1, keepdim=True).detach()).mm(A)
            X_t_new = (X_t / torch.linalg.norm(X_t, dim=-1, keepdim=True).detach()).mm(A)
            X_s_new /= torch.linalg.norm(X_s_new, dim=-1, keepdim=True).detach()
            X_t_new /= torch.linalg.norm(X_t_new, dim=-1, keepdim=True).detach()
        else:
            X_s_new = X_s.mm(A)
            X_t_new = X_t.mm(A)
        return X_s_new, X_t_new


def sym_norm_graph(Graph):
    shape = Graph.shape
    dense = Graph.to_dense()
    D = torch.sum(dense, dim=1).float()
    D[D == 0.] = 1.
    D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
    dense = dense / D_sqrt
    dense = dense / D_sqrt.t()
    index = dense.nonzero(as_tuple=False)
    data = dense[dense >= 1e-9]
    assert len(index) == len(data)
    Graph = torch.sparse_coo_tensor(index.t(), data, shape, dtype=torch.float)
    Graph = Graph.coalesce()
    return Graph


def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape), dtype=torch.float32)


class BipartitleGraph:
    def __init__(self, args, dataset):
        self.graph = self.construct_graph(args, dataset)
        indices = torch.arange(dataset.num_users).long().cuda()
        self.R = self.graph.index_select(dim=0, index=indices)
        indices = torch.arange(dataset.num_items).long().cuda() + dataset.num_users
        self.R = self.R.index_select(dim=1, index=indices)
        
    def construct_graph(self, args, dataset):
        f_Graph = os.path.join("modeling", "backbone", "crafts", args.dataset, f"Graph.pkl")
        sucflg, Graph = load_pkls(f_Graph)
        if sucflg:
            return Graph.cuda()
        
        config = yaml.load(open(os.path.join(args.DATA_DIR, args.dataset, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
        if "large" in config and config["large"] == True:
            Graph = self._construct_large_graph(dataset)
        else:
            Graph = self._construct_small_graph(dataset)
        dump_pkls((Graph, f_Graph))
        return Graph.cuda()
    
    def _construct_small_graph(self, dataset):
        user_dim = torch.LongTensor(dataset.train_pairs[:, 0].cpu())
        item_dim = torch.LongTensor(dataset.train_pairs[:, 1].cpu())

        first_sub = torch.stack([user_dim, item_dim + dataset.num_users])
        second_sub = torch.stack([item_dim + dataset.num_users, user_dim])
        index = torch.cat([first_sub, second_sub], dim=1)
        data = torch.ones(index.size(-1)).int()
        Graph = torch.sparse_coo_tensor(index, data,
                                            torch.Size([dataset.num_users + dataset.num_items, dataset.num_users + dataset.num_items]), dtype=torch.int)
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
            [dataset.num_users + dataset.num_items, dataset.num_users + dataset.num_items]), dtype=torch.float)
        Graph = Graph.coalesce()
        return Graph

    def _construct_large_graph(self, dataset):
        adj_mat = sp.dok_matrix((dataset.num_users + dataset.num_items, dataset.num_users + dataset.num_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        train_pairs = dataset.train_pairs.numpy()
        UserItemNet  = csr_matrix((np.ones(len(train_pairs)), (train_pairs[:, 0], train_pairs[:, 1])), shape=(dataset.num_users, dataset.num_items))
        R = UserItemNet.tolil()
        adj_mat[:dataset.num_users, dataset.num_users:] = R
        adj_mat[dataset.num_users:, :dataset.num_users] = R.T
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
