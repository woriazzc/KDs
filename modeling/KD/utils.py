import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


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
        super(Expert, self).__init__()
        mlps = []
        for i in range(len(dims) - 2):
            mlps.append(nn.Linear(dims[i], dims[i + 1]))
            mlps.append(nn.ReLU())
        mlps.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*mlps)

    def forward(self, x):
        return self.mlp(x)


class nosepExpert(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, is_teacher=False, norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.norm = norm
        if is_teacher:
            self.experts = nn.ModuleList([nn.Linear(self.input_dim, self.output_dim) for i in range(self.num_experts)])
        else:
            expert_dims = [self.input_dim, (self.input_dim + self.output_dim) // 2, self.output_dim]
            self.experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])

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
