import os
import math
import mlflow
import random
import numpy as np
from copy import deepcopy

from torch_cluster import knn
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Projector, pca, load_pkls, dump_pkls, self_loop_graph, CKA, info_abundance, log_mmd, TCA, MMD_loss
from .base_model import BaseKD4Rec, BaseKD4CTR
from .baseline import DE


class CPD(DE):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)

        self.step = 0

        self.alpha = args.cpd_alpha
        self.sample_type = args.cpd_sample_type
        self.reg_type = args.cpd_reg_type
        self.guide_type = args.cpd_guide_type
        self.Q = args.cpd_Q

        if self.args.ablation:
            self.sample_type = "none"
            self.reg_type = "none"
        
        if self.reg_type == "list":
            self.tau_ce = args.cpd_tau_ce
        
        if self.sample_type == "rank":
            # For interesting item
            self.T = args.cpd_T
            self.mxK = min(args.cpd_mxK, self.num_items)
            ranking_list = torch.exp(-(torch.arange(self.mxK) + 1) / self.T)
            self.ranking_mat = ranking_list.repeat(self.num_users, 1)
            if self.guide_type == "teacher":
                self.topk_dict = self.get_topk_dict(self.teacher)

    def get_topk_dict(self, model):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = model.get_all_ratings()
            # TODO: delete it ??
            # train_pairs = self.dataset.train_pairs
            # inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            _, topk_dict = torch.topk(inter_mat, self.mxK, dim=-1)
        return topk_dict.type(torch.LongTensor)
    
    def ce_loss(self, S, T):
        T_probs = torch.softmax(T / self.tau_ce, dim=-1)
        return F.cross_entropy(S / self.tau_ce, T_probs, reduction='sum')

    def ranking_loss(self, S, T):
        _, idx_col = torch.sort(T, descending=True)
        idx_row = torch.arange(T.size(0)).unsqueeze(1).cuda()
        S = S[idx_row, idx_col]
        above = S.sum(1, keepdims=True)
        below = S.flip(-1).exp().cumsum(1)
        below = below.log().sum(1, keepdims=True)
        loss = -(above - below)
        margin = (torch.arange(self.Q) + 1).log().sum().cuda()
        loss[loss < margin] = 0.
        return loss.sum()
    
    def ranking_loss2(self, S, T):
        _, idx_col = torch.sort(T, descending=True)
        idx_row = torch.arange(S.size(0)).cuda().reshape(-1, 1, 1)
        idx_dim2 = torch.arange(S.size(1)).cuda().reshape(1, -1, 1)
        S = S[idx_row, idx_dim2, idx_col]
        above = S.sum(-1, keepdims=True)
        below = S.flip(-1).exp().cumsum(-1)
        below = below.log().sum(-1, keepdims=True)
        loss = -(above - below)
        margin = (torch.arange(S.shape[-1]) + 1).log().sum().cuda()
        loss[loss < margin] = 0.
        return loss.mean(1).sum()

    def do_something_in_each_epoch(self, epoch):
        self.current_T = self.end_T * self.anneal_size * ((1. / self.anneal_size) ** (epoch / self.max_epoch))
        self.current_T = max(self.current_T, self.end_T)

        if self.args.ablation:
            return
        
        with torch.no_grad():
            if self.sample_type == "rank":
                if self.guide_type == "student":
                    self.topk_dict = self.get_topk_dict(self.student)
                self.sampled_items = torch.zeros((self.num_users, self.Q), dtype=torch.long)
                samples = torch.multinomial(self.ranking_mat, self.Q, replacement=False)
                for user in range(self.num_users):
                    self.sampled_items[user] = self.topk_dict[user][samples[user]]
                self.sampled_items = self.sampled_items.cuda()
            elif self.sample_type == "uniform":
                self.sampled_items = torch.from_numpy(np.random.choice(self.num_items, size=(self.num_users, self.Q), replace=True)).cuda()

    def get_features(self, batch_entity, is_user, is_teacher, detach=False):
        size = batch_entity.size()
        batch_entity = batch_entity.reshape(-1)
        
        if is_user:
            s = self.student.get_user_embedding(batch_entity)
            t = self.teacher.get_user_embedding(batch_entity)

            experts = self.user_experts
            selection_net = self.user_selection_net
        else:
            s = self.student.get_item_embedding(batch_entity)
            t = self.teacher.get_item_embedding(batch_entity)
            
            experts = self.item_experts
            selection_net = self.item_selection_net
        
        if is_teacher:
            return t.reshape(*size, -1)
        
        selection_dist = selection_net(t) 			# batch_size x num_experts

        if self.num_experts == 1:
            selection_result = 1.
        else:
            # Expert Selection
            g = torch.distributions.Gumbel(0, 1).sample(selection_dist.size()).cuda()
            eps = 1e-10 										# for numerical stability
            selection_dist = selection_dist + eps
            selection_dist = self.sm((selection_dist.log() + g) / self.current_T)

            selection_dist = torch.unsqueeze(selection_dist, 1)					# batch_size x 1 x num_experts
            selection_result = selection_dist.repeat(1, self.teacher_dim, 1)			# batch_size x teacher_dims x num_experts
        
        if detach:
            s = s.detach()

        expert_outputs = [experts[i](s).unsqueeze(-1) for i in range(self.num_experts)] 		# s -> t
        expert_outputs = torch.cat(expert_outputs, -1)							# batch_size x teacher_dims x num_experts

        expert_outputs = expert_outputs * selection_result						# batch_size x teacher_dims x num_experts
        expert_outputs = expert_outputs.sum(2)
        return expert_outputs.reshape(*size, -1)

    def get_DE_loss(self, batch_entity, is_user):
        T_feas = self.get_features(batch_entity, is_user=is_user, is_teacher=True)
        S_feas = self.get_features(batch_entity, is_user=is_user, is_teacher=False)

        DE_loss = ((T_feas - S_feas) ** 2).sum(-1).sum()
        return DE_loss

    def get_reg(self, batch_user, batch_pos_item, batch_neg_item):
        if self.reg_type == "pair":
            if self.sample_type == "batch":
                post_u_feas = self.get_features(batch_user, is_user=True, is_teacher=False, detach=True)
                post_pos_feas = self.get_features(batch_pos_item, is_user=False, is_teacher=False, detach=True)
                post_neg_feas = self.get_features(batch_neg_item, is_user=False, is_teacher=False, detach=True)
                post_score_1 = (post_u_feas * post_pos_feas).sum(-1).unsqueeze(-1)  # bs, 1
                post_score_2 = torch.bmm(post_neg_feas, post_u_feas.unsqueeze(-1)).squeeze(-1)    # bs, num_ns

                pre_u_feas = self.student.get_user_embedding(batch_user).detach()
                pre_pos_feas = self.student.get_item_embedding(batch_pos_item).detach()
                pre_neg_feas = self.student.get_item_embedding(batch_neg_item).detach()
                pre_score_1 = (pre_u_feas * pre_pos_feas).sum(-1).unsqueeze(-1)
                pre_score_2 = torch.bmm(pre_neg_feas, pre_u_feas.unsqueeze(-1)).squeeze(-1)
                # reg = F.relu(-(post_score_1 - post_score_2) * (pre_score_1 - pre_score_2)).mean(-1).sum()
                reg = -F.logsigmoid((post_score_1 - post_score_2) * (pre_score_1 - pre_score_2)).mean(-1).sum()
            elif self.sample_type in ["uniform", "rank"]:
                batch_item_1, batch_item_2 = self.sampled_items[batch_user, 0], self.sampled_items[batch_user, 1]
                post_u_feas = self.get_features(batch_user, is_user=True, is_teacher=False, detach=True)
                post_feas_1 = self.get_features(batch_item_1, is_user=False, is_teacher=False, detach=True)
                post_feas_2 = self.get_features(batch_item_2, is_user=False, is_teacher=False, detach=True)
                post_score_1 = (post_u_feas * post_feas_1).sum(-1)  # bs
                post_score_2 = (post_u_feas * post_feas_2).sum(-1)  # bs

                pre_u_feas = self.student.get_user_embedding(batch_user).detach()
                pre_feas_1 = self.student.get_item_embedding(batch_item_1).detach()
                pre_feas_2 = self.student.get_item_embedding(batch_item_2).detach()
                pre_score_1 = (pre_u_feas * pre_feas_1).sum(-1)   # bs
                pre_score_2 = (pre_u_feas * pre_feas_2).sum(-1)   # bs
                # reg = F.relu(-(post_score_1 - post_score_2) * (pre_score_1 - pre_score_2)).sum()
                reg = -F.logsigmoid((post_score_1 - post_score_2) * (pre_score_1 - pre_score_2)).sum()
            
            else: raise NotImplementedError

        elif self.reg_type == "list":
            Q_items = self.sampled_items[batch_user].type(torch.LongTensor).cuda()
            post_u_feas = self.get_features(batch_user, is_user=True, is_teacher=False, detach=True)     # bs, S_dim
            post_i_feas = self.get_features(Q_items, is_user=False, is_teacher=False, detach=True)    # bs, Q, S_dim
            post_Q_logits = torch.bmm(post_i_feas, post_u_feas.unsqueeze(-1)).squeeze(-1)    # bs, Q
            
            pre_u_feas = self.student.get_user_embedding(batch_user).detach()   # bs, S_dim
            pre_i_feas = self.student.get_item_embedding(Q_items).detach()   # bs, Q, S_dim
            pre_Q_logits = torch.bmm(pre_i_feas, pre_u_feas.unsqueeze(-1)).squeeze(-1)    # bs, Q
            reg = self.ce_loss(post_Q_logits, pre_Q_logits)
        else:
            raise NotImplementedError
        return reg

    @torch.no_grad()
    def log_incon(self, batch_user):
        sampled_items = torch.from_numpy(np.random.choice(self.num_items, size=(batch_user.size(0), 2), replace=True)).cuda()
        batch_item_1, batch_item_2 = sampled_items[:, 0], sampled_items[:, 1]
        post_u_feas = self.get_features(batch_user, is_user=True, is_teacher=False, detach=True)
        post_feas_1 = self.get_features(batch_item_1, is_user=False, is_teacher=False, detach=True)
        post_feas_2 = self.get_features(batch_item_2, is_user=False, is_teacher=False, detach=True)
        post_score_1 = (post_u_feas * post_feas_1).sum(-1)  # bs
        post_score_2 = (post_u_feas * post_feas_2).sum(-1)  # bs

        pre_u_feas = self.student.get_user_embedding(batch_user).detach()
        pre_feas_1 = self.student.get_item_embedding(batch_item_1).detach()
        pre_feas_2 = self.student.get_item_embedding(batch_item_2).detach()
        pre_score_1 = (pre_u_feas * pre_feas_1).sum(-1)   # bs
        pre_score_2 = (pre_u_feas * pre_feas_2).sum(-1)   # bs
        incon = ((post_score_1 - post_score_2) * (pre_score_1 - pre_score_2) < 0).float().mean()
        mlflow.log_metric("incon", incon.detach().cpu().item(), self.step)
        self.step += 1

    @torch.no_grad()
    def log_group_incon(self, batch_user):
        G = 5
        top_items = self.topk_dict.cuda()[batch_user]
        bs = top_items.size(1) // G
        for i in range(G):
            for j in range(i, G):
                batch_item_1 = top_items[torch.arange(batch_user.size(0)).cuda(), torch.randint(bs * i, bs * (i + 1), (batch_user.size(0),)).cuda()]
                batch_item_2 = top_items[torch.arange(batch_user.size(0)).cuda(), torch.randint(bs * j, bs * (j + 1), (batch_user.size(0),)).cuda()]
                post_u_feas = self.get_features(batch_user, is_user=True, is_teacher=False, detach=True)
                post_feas_1 = self.get_features(batch_item_1, is_user=False, is_teacher=False, detach=True)
                post_feas_2 = self.get_features(batch_item_2, is_user=False, is_teacher=False, detach=True)
                post_score_1 = (post_u_feas * post_feas_1).sum(-1)  # bs
                post_score_2 = (post_u_feas * post_feas_2).sum(-1)  # bs

                pre_u_feas = self.student.get_user_embedding(batch_user).detach()
                pre_feas_1 = self.student.get_item_embedding(batch_item_1).detach()
                pre_feas_2 = self.student.get_item_embedding(batch_item_2).detach()
                pre_score_1 = (pre_u_feas * pre_feas_1).sum(-1)   # bs
                pre_score_2 = (pre_u_feas * pre_feas_2).sum(-1)   # bs
                incon = ((post_score_1 - post_score_2) * (pre_score_1 - pre_score_2) < 0).float().mean()
                mlflow.log_metric(f"incon_{i}_{j}", incon.detach().cpu().item(), self.step)
        self.step += 1
    
    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        DE_loss_user = self.get_DE_loss(batch_user.unique(), is_user=True)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), is_user=False)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), is_user=False)
        DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5
        if self.args.ablation: reg = 0.
        else: reg = self.get_reg(batch_user, batch_pos_item, batch_neg_item)

        if self.args.verbose:
            self.log_group_incon(batch_user)

        loss = DE_loss + self.alpha * reg
        return loss


class NKD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "nkd"

        self.num_experts = args.nkd_num_experts
        self.strategy = args.nkd_strategy
        self.alpha = args.nkd_alpha
        self.K = args.nkd_K
        self.dropout_rate = args.nkd_dropout_rate
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.S_user_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=True, dropout_rate=self.dropout_rate)
        self.S_item_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=True, dropout_rate=self.dropout_rate)

        all_u, all_i = self.teacher.get_all_embedding()
        self.nearestK_u, self.nearestK_i = self.get_nearest_K(all_u, all_i, self.K)

    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def _KNN(self, embs, K):
        with torch.no_grad():
            embs = pca(embs, 150)
            topk_indices = knn(embs, embs, k=K+1)[1].reshape(-1, K + 1)
        return topk_indices[:, 1:].cuda()

    def get_nearest_K(self, all_u, all_i, K):
        f_nearestK_u = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.backbone, self.model_name, f"nearest_{K}_u.pkl")
        f_nearestK_i = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.backbone, self.model_name, f"nearest_{K}_i.pkl")
        sucflg, nearestK_u, nearestK_i = load_pkls(f_nearestK_u, f_nearestK_i)
        if not sucflg:
            nearestK_u = self._KNN(all_u, K)
            nearestK_i = self._KNN(all_i, K)
            dump_pkls((nearestK_u, f_nearestK_u), (nearestK_i, f_nearestK_i))
        return nearestK_u, nearestK_i
    
    def get_features(self, batch_entity, is_user):
        N = batch_entity.shape[0]
        if is_user:
            T = self.teacher.get_user_embedding(batch_entity)
            S = self.student.get_user_embedding(batch_entity)
            experts = self.S_user_experts
        else:
            T = self.teacher.get_item_embedding(batch_entity)
            S = self.student.get_item_embedding(batch_entity)
            experts = self.S_item_experts
        
        rnd_choice = torch.randint(0, self.K, (N, ), device='cuda')
        if is_user:
            neighborsS = self.student.get_user_embedding(self.nearestK_u[batch_entity, rnd_choice])     # bs, S_dim
            neighborsT = self.teacher.get_user_embedding(self.nearestK_u[batch_entity, rnd_choice])     # bs, T_dim
        else:
            neighborsS = self.student.get_item_embedding(self.nearestK_i[batch_entity, rnd_choice])     # bs, S_dim
            neighborsT = self.teacher.get_item_embedding(self.nearestK_i[batch_entity, rnd_choice])     # bs, T_dim

        if self.strategy == 'soft':
            rndS = torch.rand_like(S, device='cuda') * self.alpha * 2
            rndT = torch.rand_like(T, device='cuda') * self.alpha * 2
            S = rndS * S + (1. - rndS) * neighborsS     # bs, S_dim
            T = rndT * T + (1. - rndT) * neighborsT     # bs, T_dim
        elif self.strategy == 'hard':
            rndS = torch.rand_like(S, device='cuda')
            rndT = torch.rand_like(T, device='cuda')
            S = torch.where(rndS < self.alpha, S, neighborsS)   # bs, S_dim
            T = torch.where(rndT < self.alpha, T, neighborsT)   # bs, T_dim
        elif self.strategy == 'mix':
            S = self.alpha * S + (1. - self.alpha) * neighborsS
            T = self.alpha * T + (1. - self.alpha) * neighborsT
        elif self.strategy == 'randmix':
            rndS = random.random() * self.alpha * 2
            rndT = random.random() * self.alpha * 2
            S = rndS * S + (1. - rndS) * neighborsS
            T = rndT * T + (1. - rndT) * neighborsT
        elif self.strategy == 'batchmix':
            rndS = torch.rand((N, 1), device='cuda')
            rndT = torch.rand((N, 1), device='cuda')
            S = rndS * S + (1. - rndS) * neighborsS
            T = rndT * T + (1. - rndT) * neighborsT
        elif self.strategy == 'layermix':
            rndS = torch.rand((1, S.size(1)), device='cuda')
            rndT = torch.rand((1, T.size(1)), device='cuda')
            S = rndS * S + (1. - rndS) * neighborsS
            T = rndT * T + (1. - rndT) * neighborsT
        elif self.strategy == 'allmix':
            rndS = torch.rand((1, S.size(1)), device='cuda') * torch.rand((N, 1), device='cuda')
            rndT = torch.rand((1, T.size(1)), device='cuda') * torch.rand((N, 1), device='cuda')
            S = rndS * S + (1. - rndS) * neighborsS
            T = rndT * T + (1. - rndT) * neighborsT
        elif self.strategy == 'hardmix':
            rndS = random.random()
            if rndS >= self.alpha:
                S = neighborsS
            rndT = random.random()
            if rndT >= self.alpha:
                T = neighborsT
        else:
            raise NotImplementedError
        
        S = experts(S)
        return T, S
    
    def get_DE_loss(self, batch_entity, is_user):
        T_feas, S_feas = self.get_features(batch_entity, is_user=is_user)

        norm_T = T_feas.pow(2).sum(-1, keepdim=True).pow(1. / 2)
        T_feas = T_feas.div(norm_T)
        cos_theta = (T_feas * S_feas).sum(-1, keepdim=True)
        G_diff = 1. - cos_theta
        DE_loss = G_diff.sum()
        return DE_loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        DE_loss_user = self.get_DE_loss(batch_user.unique(), is_user=True)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), is_user=False)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), is_user=False)

        DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5
        return DE_loss


class GraphD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "graphd"
        self.ablation = getattr(args, "ablation", False)

        self.num_experts = args.graphd_num_experts
        self.alpha = args.graphd_alpha
        self.K = args.graphd_K
        self.keep_prob = args.graphd_keep_prob
        self.dropout_rate = args.graphd_dropout_rate
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.S_user_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=True, dropout_rate=self.dropout_rate)
        self.S_item_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=True, dropout_rate=self.dropout_rate)

        if self.ablation:
            self.Graph_u, self.Graph_i = None, None
        else:
            all_u, all_i = self.teacher.get_all_embedding()
            self.nearestK_u, self.nearestK_i = self.get_nearest_K(all_u, all_i, self.K)
            self.Graph_u = self.construct_knn_graph(self.nearestK_u)
            self.Graph_i = self.construct_knn_graph(self.nearestK_i)
    
    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def _KNN(self, embs, K):
        with torch.no_grad():
            embs = pca(embs, 150)
            topk_indices = knn(embs, embs, k=K+1)[1].reshape(-1, K + 1)
        return topk_indices[:, 1:].cuda()

    def get_nearest_K(self, all_u, all_i, K):
        f_nearestK_u = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.backbone, self.model_name, f"nearest_{K}_u.pkl")
        f_nearestK_i = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.backbone, self.model_name, f"nearest_{K}_i.pkl")
        sucflg, nearestK_u, nearestK_i = load_pkls(f_nearestK_u, f_nearestK_i)
        if not sucflg:
            nearestK_u = self._KNN(all_u, K)
            nearestK_i = self._KNN(all_i, K)
            dump_pkls((nearestK_u, f_nearestK_u), (nearestK_i, f_nearestK_i))
        return nearestK_u, nearestK_i
    
    def construct_knn_graph(self, neighbor_id):
        N, K = neighbor_id.shape
        row = torch.arange(N).repeat(K, 1).T.reshape(-1).cuda()
        col = neighbor_id.reshape(-1)
        index = torch.stack([row, col])
        data = torch.ones(index.size(-1)).cuda() / K
        Graph = torch.sparse_coo_tensor(index, data,
                                            torch.Size([N, N]), dtype=torch.float)
        Graph = Graph.coalesce()
        return Graph.cuda()

    def _dropout_graph(self, Graph):
        size = Graph.size()
        assert size[0] == size[1]
        index = Graph.indices().t()
        values = Graph.values()
        random_index = torch.rand(len(values)) + self.keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / self.keep_prob

        self_loop_idx = torch.stack([torch.arange(size[0]), torch.arange(size[0])], dim=1).cuda()
        self_loop_data = torch.ones(self_loop_idx.size(0)).cuda()

        rndS = random.random() * self.alpha * 2
        rndT = random.random() * self.alpha * 2
        # rndS = torch.randn(1, device='cuda') * 0.05 + self.alpha
        # rndT = torch.randn(1, device='cuda') * 0.05 + self.alpha
        valuesS = torch.cat([values * (1. - rndS), self_loop_data * rndS])
        valuesT = torch.cat([values * (1. - rndT), self_loop_data * rndT])

        index = torch.cat([index, self_loop_idx], dim=0)

        droped_GraphS = torch.sparse_coo_tensor(index.t(), valuesS, size, dtype=torch.float)
        droped_GraphT = torch.sparse_coo_tensor(index.t(), valuesT, size, dtype=torch.float)
        return droped_GraphS, droped_GraphT

    def get_features(self, batch_entity, is_user):
        if is_user:
            T = self.teacher.user_emb.weight
            S = self.student.user_emb.weight
            experts = self.S_user_experts
            Graph = self.Graph_u
        else:
            T = self.teacher.item_emb.weight
            S = self.student.item_emb.weight
            experts = self.S_item_experts
            Graph = self.Graph_i
        
        if not self.ablation:
            droped_GraphS, droped_GraphT = self._dropout_graph(Graph)
            
            S = torch.sparse.mm(droped_GraphS, S)
            T = torch.sparse.mm(droped_GraphT, T)
        T = T[batch_entity]
        S = S[batch_entity]
        S = experts(S)
        return T, S
    
    def get_DE_loss(self, batch_entity, is_user):
        T_feas, S_feas = self.get_features(batch_entity, is_user)

        norm_T = T_feas.pow(2).sum(-1, keepdim=True).pow(1. / 2)
        T_feas = T_feas.div(norm_T)
        cos_theta = (T_feas * S_feas).sum(-1, keepdim=True)
        G_diff = 1. - cos_theta
        DE_loss = G_diff.sum()
        return DE_loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        DE_loss_user = self.get_DE_loss(batch_user.unique(), True)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), False)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), False)

        DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5
        return DE_loss


class FilterD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "filterd"

        self.num_experts = args.filterd_num_experts
        self.alpha = args.filterd_alpha
        self.beta = args.filterd_beta
        self.eig_ratio = args.filterd_eig_ratio
        self.K = args.filterd_K
        self.dropout_rate = args.filterd_dropout_rate
        self.filter_type = args.filterd_type
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.S_user_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=True, dropout_rate=self.dropout_rate)
        self.S_item_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=True, dropout_rate=self.dropout_rate)

        all_u, all_i = self.teacher.get_all_embedding()
        nearestK_u, nearestK_i = self.get_nearest_K(all_u, all_i, self.K)
        self.filter_u = self.construct_knn_filter(nearestK_u, filter_type=self.filter_type, entity_type='u')
        self.filter_i = self.construct_knn_filter(nearestK_i, filter_type=self.filter_type, entity_type='i')
    
    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def _KNN(self, embs, K):
        with torch.no_grad():
            embs = pca(embs, 150)
            topk_indices = knn(embs, embs, k=K+1)[1].reshape(-1, K + 1)
        return topk_indices[:, 1:].cuda()

    def get_nearest_K(self, all_u, all_i, K):
        f_nearestK_u = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.backbone, self.model_name, f"nearest_{K}_u.pkl")
        f_nearestK_i = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.backbone, self.model_name, f"nearest_{K}_i.pkl")
        sucflg, nearestK_u, nearestK_i = load_pkls(f_nearestK_u, f_nearestK_i)
        if not sucflg:
            nearestK_u = self._KNN(all_u, K)
            nearestK_i = self._KNN(all_i, K)
            dump_pkls((nearestK_u, f_nearestK_u), (nearestK_i, f_nearestK_i))
        return nearestK_u, nearestK_i
    
    def construct_knn_filter(self, neighbor_id, filter_type, entity_type):
        N, K = neighbor_id.shape
        smooth_dim = int(N * self.eig_ratio)
        f_smooth_values = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.backbone, self.model_name, f"{filter_type}_values_{entity_type}_{smooth_dim}.pkl")
        f_smooth_vectors = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.backbone, self.model_name, f"{filter_type}_vectors_{entity_type}_{smooth_dim}.pkl")
        sucflg, smooth_values, smooth_vectors = load_pkls(f_smooth_values, f_smooth_vectors)
        if sucflg:
            filter = (smooth_vectors * self.weight_feature(smooth_values)).mm(smooth_vectors.t())
            return filter.cuda()
        
        row = torch.arange(N).repeat(K, 1).T.reshape(-1).cuda()
        col = neighbor_id.reshape(-1)
        index = torch.stack([row, col], dim=1)
        data = torch.ones(index.size(0)).cuda() / K

        self_loop_idx = torch.stack([torch.arange(N), torch.arange(N)], dim=1).cuda()
        self_loop_data = torch.ones(self_loop_idx.size(0)).cuda()

        data = torch.cat([data * (1. - self.alpha), self_loop_data * self.alpha])
        index = torch.cat([index, self_loop_idx], dim=0)

        Graph = torch.sparse_coo_tensor(index.t(), data, torch.Size([N, N]), dtype=torch.float)
        Graph = (Graph.t() + Graph) / 2.
        Graph = self._sym_normalize(Graph)

        if self.eig_ratio <= 0.3:
            smooth_values, smooth_vectors = torch.lobpcg(Graph, k=smooth_dim, largest=(filter_type == "smooth"), niter=5)
        else:
            assert filter_type == "smooth", "rough filter is only supported when eig_ratio <= 0.3"
            smooth_vectors, smooth_values, _ = torch.svd_lowrank(Graph, q=smooth_dim, niter=10)
        dump_pkls((smooth_values, f_smooth_values), (smooth_vectors, f_smooth_vectors))
        filter = (smooth_vectors * self.weight_feature(smooth_values)).mm(smooth_vectors.t())
        return filter.cuda()
    
    def weight_feature(self, value):
        # return torch.exp(self.beta * (value - value.max())).reshape(1, -1)
        return torch.clip(self.beta * (value - value.max()) + 1., 0.).reshape(1, -1)
        # return torch.clip(1. - self.beta * (value - value.min()), 0.).reshape(1, -1)
    
    def _sym_normalize(self, Graph):
        dense = Graph.to_dense().cpu()
        D = torch.sum(dense, dim=1).float()
        D[D == 0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense / D_sqrt.t()
        index = dense.nonzero(as_tuple=False)
        data = dense[dense != 0]
        assert len(index) == len(data)
        Graph = torch.sparse_coo_tensor(index.t(), data, torch.Size(Graph.size()), dtype=torch.float)
        return Graph.coalesce().cuda()

    def get_features(self, batch_entity, is_user):
        if is_user:
            T = self.teacher.user_emb.weight
            S = self.student.user_emb.weight
            experts = self.S_user_experts
            Graph = self.filter_u
        else:
            T = self.teacher.item_emb.weight
            S = self.student.item_emb.weight
            experts = self.S_item_experts
            Graph = self.filter_i
        
        filtered_S = torch.sparse.mm(Graph, S)
        filtered_T = torch.sparse.mm(Graph, T)
        filtered_T = filtered_T[batch_entity]
        filtered_S = filtered_S[batch_entity]
        filtered_S = experts(filtered_S)

        return filtered_S, filtered_T
    
    def cal_FD_loss(self, T_feas, S_feas):
        norm_T = T_feas.pow(2).sum(-1, keepdim=True).pow(1. / 2)
        T_feas = T_feas.div(norm_T)
        cos_theta = (T_feas * S_feas).sum(-1, keepdim=True)
        G_diff = 1. - cos_theta
        FD_loss = G_diff.sum()
        return FD_loss
    
    def get_DE_loss(self, batch_entity, is_user):
        filtered_S, filtered_T = self.get_features(batch_entity, is_user)
        DE_loss = self.cal_FD_loss(filtered_T, filtered_S)
        return DE_loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        DE_loss_user = self.get_DE_loss(batch_user.unique(), True)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), False)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), False)

        DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5
        return DE_loss


class FD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "fd"

        self.num_experts = args.fd_num_experts
        self.dropout_rate = args.fd_dropout_rate
        self.alpha = args.fd_alpha
        self.beta = args.fd_beta
        self.eig_ratio = args.fd_eig_ratio
        self.num_anchors = args.fd_num_anchors
        self.K = args.fd_K
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.norm = False
        self.S_user_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=False, dropout_rate=self.dropout_rate)
        self.S_item_experts = Projector(self.student_dim, self.teacher_dim, self.num_experts, norm=False, dropout_rate=self.dropout_rate)

        all_u, all_i = self.teacher.get_all_embedding()
        nearestK_u, nearestK_i = self.get_nearest_K(all_u, all_i, self.K)
        self.filter_u = self.construct_knn_filter(nearestK_u, entity_type='u')
        self.filter_i = self.construct_knn_filter(nearestK_i, entity_type='i')

        self.anchor_filters_u = self.construct_anchor_filters(entity_type='u')
        self.anchor_filters_i = self.construct_anchor_filters(entity_type='i')
        self.global_step = 0
    
    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def _KNN(self, embs, K):
        with torch.no_grad():
            embs = pca(embs, 150)
            topk_indices = knn(embs, embs, k=K+1)[1].reshape(-1, K + 1)
        return topk_indices[:, 1:].cuda()

    def get_nearest_K(self, all_u, all_i, K):
        f_nearestK_u = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.backbone, self.model_name, f"nearest_{K}_u.pkl")
        f_nearestK_i = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.backbone, self.model_name, f"nearest_{K}_i.pkl")
        sucflg, nearestK_u, nearestK_i = load_pkls(f_nearestK_u, f_nearestK_i)
        if not sucflg:
            nearestK_u = self._KNN(all_u, K)
            nearestK_i = self._KNN(all_i, K)
            dump_pkls((nearestK_u, f_nearestK_u), (nearestK_i, f_nearestK_i))
        return nearestK_u, nearestK_i
    
    def construct_knn_filter(self, neighbor_id, entity_type):
        N, K = neighbor_id.shape
        smooth_dim = int(N * self.eig_ratio)
        f_smooth_values = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.backbone, self.model_name, f"smooth_values_{entity_type}_{smooth_dim}.pkl")
        f_smooth_vectors = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.backbone, self.model_name, f"smooth_vectors_{entity_type}_{smooth_dim}.pkl")
        sucflg, smooth_values, smooth_vectors = load_pkls(f_smooth_values, f_smooth_vectors)
        if sucflg:
            filter = (smooth_vectors * self.weight_feature(smooth_values)).mm(smooth_vectors.t())
            return filter.cuda()
        
        row = torch.arange(N).repeat(K, 1).T.reshape(-1).cuda()
        col = neighbor_id.reshape(-1)
        index = torch.stack([row, col], dim=1)
        data = torch.ones(index.size(0)).cuda() / K

        self_loop_idx = torch.stack([torch.arange(N), torch.arange(N)], dim=1).cuda()
        self_loop_data = torch.ones(self_loop_idx.size(0)).cuda()

        data = torch.cat([data * (1. - self.alpha), self_loop_data * self.alpha])
        index = torch.cat([index, self_loop_idx], dim=0)

        Graph = torch.sparse_coo_tensor(index.t(), data, torch.Size([N, N]), dtype=torch.float)
        Graph = (Graph.t() + Graph) / 2.
        Graph = self._sym_normalize(Graph)

        if self.eig_ratio <= 0.3:
            smooth_values, smooth_vectors = torch.lobpcg(Graph, k=smooth_dim, largest=True, niter=5)
        else:
            smooth_vectors, smooth_values, _ = torch.svd_lowrank(Graph, q=smooth_dim, niter=10)
        dump_pkls((smooth_values, f_smooth_values), (smooth_vectors, f_smooth_vectors))
        filter = (smooth_vectors * self.weight_feature(smooth_values)).mm(smooth_vectors.t())
        return filter.cuda()
    
    def construct_anchor_filters(self, entity_type):
        if self.num_anchors == 0: return []
        smooth_dim = self.num_users if entity_type == 'u' else self.num_items
        f_all_values = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.backbone, self.model_name, f"smooth_values_{entity_type}_{smooth_dim}.pkl")
        f_all_vectors = os.path.join("modeling", "KD", "crafts", self.args.dataset, self.args.backbone, self.model_name, f"smooth_vectors_{entity_type}_{smooth_dim}.pkl")
        sucflg, all_values, all_vectors = load_pkls(f_all_values, f_all_vectors)
        assert sucflg, "Can't find eig decomposition results."
        anchor_filters = []
        blk = smooth_dim // self.num_anchors
        for i in range(self.num_anchors):
            vectors = all_vectors[:, blk * i:blk * (i + 1)]
            filter = vectors.mm(vectors.t())
            anchor_filters.append(filter.cuda())
        # anchor_filters.append(self.filter_u if entity_type == 'u' else self.filter_i)
        return anchor_filters
    
    def weight_feature(self, value):
        ## exp
        # return torch.exp(self.beta * (value - value.max())).reshape(1, -1)
        ## linear
        return torch.clip(self.beta * (value - value.max()) + 1., 0.).reshape(1, -1)
        ## reverse linear
        # return torch.clip(1. - self.beta * (value - value.min()), 0.).reshape(1, -1)
        ## ideal low pass
        # return torch.cat([torch.ones_like(value[:len(value) // 4]) * 1.2, 
        #                   torch.ones_like(value[len(value) // 4:len(value) // 2]), 
        #                   torch.ones_like(value[len(value) // 2:len(value) // 4 * 3]) * 0.8,
        #                   torch.ones_like(value[len(value) // 4 * 3:]) * 0.8])
    
    def _sym_normalize(self, Graph):
        dense = Graph.to_dense().cpu()
        D = torch.sum(dense, dim=1).float()
        D[D == 0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense / D_sqrt.t()
        index = dense.nonzero(as_tuple=False)
        data = dense[dense != 0]
        assert len(index) == len(data)
        Graph = torch.sparse_coo_tensor(index.t(), data, torch.Size(Graph.size()), dtype=torch.float)
        return Graph.coalesce().cuda()

    def get_features(self, batch_entity, is_user):
        if is_user:
            T = self.teacher.user_emb.weight
            S = self.student.user_emb.weight
            experts = self.S_user_experts
            Graph = self.filter_u
        else:
            T = self.teacher.item_emb.weight
            S = self.student.item_emb.weight
            experts = self.S_item_experts
            Graph = self.filter_i
        
        SP = experts(S)
        filtered_S = torch.sparse.mm(Graph, SP)
        filtered_T = torch.sparse.mm(Graph, T)
        filtered_T = filtered_T[batch_entity]
        filtered_S = filtered_S[batch_entity]
        # filtered_S = experts(filtered_S)

        if self.norm:
            norm_T = T[batch_entity].pow(2).sum(-1, keepdim=True).pow(1. / 2)
            filtered_T = filtered_T.div(norm_T)
            norm_S = SP[batch_entity].pow(2).sum(-1, keepdim=True).pow(1. / 2)
            filtered_S = filtered_S.div(norm_S)

        return filtered_S, filtered_T
    
    def cal_FD_loss(self, T_feas, S_feas):
        G_diff = ((T_feas - S_feas) ** 2).sum(-1) / 2
        FD_loss = G_diff.sum()
        return FD_loss
    
    @torch.no_grad()
    def log_anchor_loss(self, batch_entity, is_user):
        if is_user:
            T = self.teacher.user_emb.weight
            S = self.student.user_emb.weight
            experts = self.S_user_experts
            anchor_filters = self.anchor_filters_u
        else:
            T = self.teacher.item_emb.weight
            S = self.student.item_emb.weight
            experts = self.S_item_experts
            anchor_filters = self.anchor_filters_i
        
        # experts.eval()
        SP = experts(S)
        # experts.train()
        for idx, filter in enumerate(anchor_filters):
            filtered_S = torch.sparse.mm(filter, SP)
            filtered_T = torch.sparse.mm(filter, T)
            filtered_T = filtered_T[batch_entity]
            filtered_S = filtered_S[batch_entity]
            # experts.eval()
            # filtered_S = experts(filtered_S)
            # experts.train()
            if self.norm:
                norm_T = T[batch_entity].pow(2).sum(-1, keepdim=True).pow(1. / 2)
                filtered_T = filtered_T.div(norm_T)
                norm_S = SP[batch_entity].pow(2).sum(-1, keepdim=True).pow(1. / 2)
                filtered_S = filtered_S.div(norm_S)

                # norm_SP = SP[batch_entity].pow(2).sum(-1)
                # rat_S = (norm_S.squeeze() / norm_SP).mean().cpu().item()
                # mlflow.log_metric(f"norm_ASP/norm_SP_{idx}", rat_S, step=self.global_step)
                # rat_T = (norm_T.squeeze() / T[batch_entity].pow(2).sum(-1)).mean().cpu().item()
                # mlflow.log_metric(f"norm_AT/norm_T_{idx}", rat_T, step=self.global_step)
            loss = self.cal_FD_loss(filtered_T, filtered_S)
            mlflow.log_metrics({f"anchor_loss_{idx}": (loss / batch_entity.shape[0]).cpu().item()}, step=self.global_step)
        # self.global_step += 1

    def get_DE_loss(self, batch_entity, is_user):
        filtered_S, filtered_T = self.get_features(batch_entity, is_user)
        DE_loss = self.cal_FD_loss(filtered_T, filtered_S)
        if is_user: self.log_anchor_loss(batch_entity, is_user)
        return DE_loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        DE_loss_user = self.get_DE_loss(batch_user.unique(), True)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), False)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), False)

        DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5

        with torch.no_grad():
            output = self.student(batch_user, batch_pos_item, batch_neg_item)
            base_loss = self.student.get_loss(output)
            mlflow.log_metrics({f"base_loss": (base_loss / batch_user.shape[0]).cpu().item()}, step=self.global_step)
            self.global_step += 1
        return DE_loss


class KNND(GraphD):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)

    def _dropout_graph(self, Graph):
        size = Graph.size()
        assert size[0] == size[1]
        index = Graph.indices().t()
        values = Graph.values()
        random_index = torch.rand(len(values)) + self.keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / self.keep_prob

        self_loop_idx = torch.stack([torch.arange(size[0]), torch.arange(size[0])], dim=1).cuda()
        self_loop_data = torch.ones(self_loop_idx.size(0)).cuda()

        rndS = random.random() * self.alpha * 2
        rndT = random.random() * self.alpha * 2
        # rndS = torch.randn(1, device='cuda') * 0.05 + self.alpha
        # rndT = torch.randn(1, device='cuda') * 0.05 + self.alpha
        valuesS = torch.cat([values * (1. - rndS), self_loop_data * rndS])
        valuesT = torch.cat([values * (1. - rndT), self_loop_data * rndT])

        index = torch.cat([index, self_loop_idx], dim=0)

        droped_GraphS = torch.sparse_coo_tensor(index.t(), valuesS, size, dtype=torch.float)
        droped_GraphT = torch.sparse_coo_tensor(index.t(), valuesT, size, dtype=torch.float)

        droped_GraphS = (droped_GraphS.t() + droped_GraphS) / 2.
        droped_GraphT = (droped_GraphT.t() + droped_GraphT) / 2.
        return droped_GraphS, droped_GraphT


class GDCP(GraphD):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "gdcp"

        self.reg_type = args.gdcp_reg_type
        self.dec_epoch = args.gdcp_dec_epoch
        self.init_w_reg = args.gdcp_w_reg
        self.w_reg = self.init_w_reg

        if self.reg_type == "first":
            self.margin = args.gdcp_margin
        elif self.reg_type == "second":
            self.Q = args.gdcp_Q
            self.T = args.gdcp_T
            self.mxK = args.gdcp_mxK
            self.tau_ce = args.gdcp_tau_ce
            self.topk_dict = self.get_topk_dict()
            ranking_list = torch.exp(-(torch.arange(self.mxK) + 1) / self.T)
            self.ranking_mat = ranking_list.repeat(self.num_users, 1)
        elif self.reg_type == "third":
            self.tau_ce = args.gdcp_tau_ce
        else:
            raise NotImplementedError

    def get_topk_dict(self):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings()
            train_pairs = self.dataset.train_pairs
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            _, topk_dict = torch.topk(inter_mat, self.mxK, dim=-1)
        return topk_dict
    
    def ce_ranking_loss(self, S, T):
        T_probs = torch.softmax(T / self.tau_ce, dim=-1)
        return F.cross_entropy(S / self.tau_ce, T_probs, reduction='sum')

    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def do_something_in_each_epoch(self, epoch):
        if self.dec_epoch == 0:
            self.w_reg = self.init_w_reg
        else:
            self.w_reg = max(0., 1. - epoch / self.dec_epoch) * self.init_w_reg
        
        if self.reg_type == "second":
            with torch.no_grad():
                self.interesting_items = torch.zeros((self.num_users, self.Q))
                while True:
                    samples = torch.multinomial(self.ranking_mat, self.Q, replacement=False)
                    if (samples > self.mxK).sum() == 0:
                        break
                samples = samples.sort(dim=1)[0]
                for user in range(self.num_users):
                    self.interesting_items[user] = self.topk_dict[user][samples[user]]
                self.interesting_items = self.interesting_items.cuda()

    def get_features(self, batch_entity, is_user, is_reg=False):
        if is_user:
            T = self.teacher.user_emb.weight
            S = self.student.user_emb.weight
            experts = self.S_user_experts
            Graph = self.Graph_u
        else:
            T = self.teacher.item_emb.weight
            S = self.student.item_emb.weight
            experts = self.S_item_experts
            Graph = self.Graph_i
        
        droped_GraphS, droped_GraphT = self._dropout_graph(Graph)
        
        S = torch.sparse.mm(droped_GraphS, S)
        T = torch.sparse.mm(droped_GraphT, T)
        T = T[batch_entity]
        S = S[batch_entity]
        if is_reg:
            pre_S = S.detach().clone()
        else:
            pre_S = S
        # pre_S = S
        post_S = experts(pre_S)
        if is_reg:
            return pre_S, post_S
        else:
            return T, post_S
    
    def get_reg(self, batch_user, batch_pos_item, batch_neg_item):
        if self.reg_type == "first":
            post_u_feas, pre_u_feas = self.get_features(batch_user, is_user=True, is_reg=True)
            post_pos_feas, pre_pos_feas = self.get_features(batch_pos_item, is_user=False, is_reg=True)
            post_neg_feas, pre_neg_feas = self.get_features(batch_neg_item, is_user=False, is_reg=True)

            post_pos_score = (post_u_feas * post_pos_feas).sum(-1, keepdim=True)
            post_neg_score = torch.bmm(post_neg_feas, post_u_feas.unsqueeze(-1)).squeeze(-1)
            post_pos_score = post_pos_score.expand_as(post_neg_score)

            pre_pos_score = (pre_u_feas * pre_pos_feas).sum(-1, keepdim=True)
            pre_neg_score = torch.bmm(pre_neg_feas, pre_u_feas.unsqueeze(-1)).squeeze(-1)
            pre_pos_score = pre_pos_score.expand_as(pre_neg_score)
            reg = F.relu(-(post_pos_score - post_neg_score) * (pre_pos_score - pre_neg_score) - self.margin).mean(1).sum()
        elif self.reg_type == "second":
            topQ_items = self.interesting_items[batch_user].type(torch.LongTensor).cuda()
            post_u_feas, pre_u_feas = self.get_features(batch_user, is_user=True, is_reg=True)		# bs, S_dim
            post_i_feas, pre_i_feas = self.get_features(topQ_items, is_user=False, is_reg=True)		# bs, Q, S_dim
            post_topQ_logits = torch.bmm(post_i_feas, post_u_feas.unsqueeze(-1)).squeeze(-1)    # bs, Q
            pre_topQ_logits = torch.bmm(pre_i_feas, pre_u_feas.unsqueeze(-1)).squeeze(-1)    # bs, Q
            reg = self.ce_ranking_loss(post_topQ_logits, pre_topQ_logits)
        elif self.reg_type == "third":
            topQ_items = torch.cat([batch_pos_item.reshape(-1, 1), batch_neg_item, self.nearestK_u[batch_user]], dim=-1)
            post_u_feas, pre_u_feas = self.get_features(batch_user, is_user=True, is_reg=True)		# bs, S_dim
            post_i_feas, pre_i_feas = self.get_features(topQ_items, is_user=False, is_reg=True)		# bs, Q, S_dim
            post_topQ_logits = torch.bmm(post_i_feas, post_u_feas.unsqueeze(-1)).squeeze(-1)    # bs, Q
            pre_topQ_logits = torch.bmm(pre_i_feas, pre_u_feas.unsqueeze(-1)).squeeze(-1)    # bs, Q
            reg = self.ce_ranking_loss(post_topQ_logits, pre_topQ_logits)
        else:
            raise NotImplementedError
        return reg

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        DE_loss_user = self.get_DE_loss(batch_user.unique(), True)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), False)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), False)
        DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5

        if self.w_reg > 0:
            reg = self.get_reg(batch_user, batch_pos_item, batch_neg_item)
        else:
            reg = 0.
        
        loss = DE_loss + reg * self.w_reg
        return loss


class FreqD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "freqd"

        self.alpha = args.freqd_alpha
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.S_user_experts = Projector(self.student_dim, self.teacher_dim, 1, norm=True)
        self.S_item_experts = Projector(self.student_dim, self.teacher_dim, 1, norm=True)

        self.filter = self.construct_filter(self.alpha)

    def construct_filter(self, alpha):
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
        filter = self_loop_graph(Graph.size(0)) * (1. - alpha) + Graph * alpha
        return filter.cuda()

    def get_features(self, batch_entity, is_user):
        T = torch.cat([self.teacher.user_emb.weight, self.teacher.item_emb.weight])
        S = torch.cat([self.student.user_emb.weight, self.student.item_emb.weight])
                    
        S = torch.sparse.mm(self.filter, S)
        T = torch.sparse.mm(self.filter, T)

        if is_user:
            T = T[batch_entity]
            S = S[batch_entity]
            experts = self.S_user_experts
        else:
            T = T[batch_entity + self.num_users]
            S = S[batch_entity + self.num_users]
            experts = self.S_item_experts
        
        S = experts(S)
        return T, S
    
    def get_DE_loss(self, batch_entity, is_user):
        T_feas, S_feas = self.get_features(batch_entity, is_user)

        norm_T = T_feas.pow(2).sum(-1, keepdim=True).pow(1. / 2)
        T_feas = T_feas.div(norm_T)
        cos_theta = (T_feas * S_feas).sum(-1, keepdim=True)
        G_diff = 1. - cos_theta
        DE_loss = G_diff.sum()
        return DE_loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        DE_loss_user = self.get_DE_loss(batch_user.unique(), True)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), False)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), False)

        DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5
        return DE_loss


class PrelD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "preld"
        self.K = args.preld_K
    
    def get_features(self, batch_entity, is_user):
        if is_user:
            s = self.student.get_user_embedding(batch_entity)
            t = self.teacher.get_user_embedding(batch_entity)
        else:
            s = self.student.get_item_embedding(batch_entity)
            t = self.teacher.get_item_embedding(batch_entity)
        return s, t

    def pca_with_grad(self, X, K):
        U, S, V = torch.svd_lowrank(X, K)
        Y = U
        # Y = U.mm(torch.diag(S))
        Y = X.mm(X.T)
        return Y
    
    def pca_mse(self, s, t):
        s_red = self.pca_with_grad(s, self.K)
        t_red = self.pca_with_grad(t, self.K)
        loss = (s_red - t_red).pow(2).sum(-1).mean()
        return loss
    
    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        users = batch_user.unique()
        items = torch.cat([batch_pos_item, batch_neg_item.flatten()], 0).unique()
        su, tu = self.get_features(users, is_user=True)
        si, ti = self.get_features(items, is_user=False)
        loss_u = self.pca_mse(su, tu)
        loss_i = self.pca_mse(si, ti)
        loss = loss_u + loss_i
        # mlflow.log_metrics({"loss_u":loss_u.item(), "loss_i":loss_i.item()})
        return loss

    # def forward(self, batch_user, batch_pos_item, batch_neg_item):
    #     output = self.student(batch_user, batch_pos_item, batch_neg_item)
    #     base_loss = self.student.get_loss(output)
    #     kd_loss = self.get_loss(batch_user, batch_pos_item, batch_neg_item)
    #     loss = kd_loss
    #     return loss, base_loss.detach(), kd_loss.detach()


class HetD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "hetd"
        self.beta = args.hetd_beta
        self.gamma = args.hetd_gamma
        self.lmbda = args.lmbda
        self.verbose = args.verbose

        student_penultimate_dim = self.student._penultimate_dim
        if isinstance(self.teacher._penultimate_dim, int):
            # For one-stream models
            teacher_penultimate_dim = self.teacher._penultimate_dim
        else:
            # For two-stream models
            cross_dim, deep_dim = self.teacher._penultimate_dim
            teacher_penultimate_dim = cross_dim + deep_dim
        self.projector = nn.Linear(student_penultimate_dim, teacher_penultimate_dim)
        self.adaptor = nn.Linear(teacher_penultimate_dim, teacher_penultimate_dim)
        self.predictor = nn.Linear(teacher_penultimate_dim, 1)
    
    # def do_something_in_each_epoch(self, epoch):
    #     if self.verbose:
    #         if epoch > 0 and not self.cka is None:
    #             print(self.cka)
    #         self.cka = None
    #         self.cnt = 0
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        if isinstance(self.teacher._penultimate_dim, tuple):
            # Two-stream models
            T_emb_cross, T_emb_deep = T_emb
            T_emb = torch.cat([T_emb_cross, T_emb_deep], dim=-1)
        S_emb = self.projector(S_emb)
        T_emb = self.adaptor(T_emb)
        T_logits = self.predictor(T_emb)
        S_pred = torch.sigmoid(self.student(data))
        loss_adaptor = F.binary_cross_entropy_with_logits(T_logits.squeeze(-1), label.squeeze(-1).float()) + self.beta * F.binary_cross_entropy_with_logits(T_logits.squeeze(-1), S_pred.detach().squeeze(-1))
        loss = (T_emb.detach() - S_emb).pow(2).sum(-1).mean() + loss_adaptor / (self.lmbda + 1e-8) * self.gamma

        # if self.verbose and self.cnt < 5:
        #     # # calculate CKA
        #     # with torch.no_grad():
        #     #     S_embs = self.student.forward_all_feature(data)
        #     #     T_embs = self.teacher.forward_all_feature(data)
        #     #     T_embs += [T_emb.detach()]
        #     #     CKA_mat = np.zeros((len(T_embs), len(S_embs)))
        #     #     for id_T, T_emb in enumerate(T_embs):
        #     #         for id_S, S_emb in enumerate(S_embs):
        #     #             CKA_mat[id_T, id_S] = CKA(T_emb, S_emb).item()
        #     #     if self.cka is None:
        #     #         self.cka = CKA_mat
        #     #     else:
        #     #         self.cka = (self.cka * self.cnt + CKA_mat) / (self.cnt + 1)
        #     #         self.cnt += 1

        #     # calculate information abundance
        #     with torch.no_grad():
        #         S_emb = self.student.forward_penultimate(data)
        #         T_emb = self.teacher.forward_penultimate(data)
        #         S_emb = S_emb.reshape(S_emb.shape[0], -1)
        #         T_emb = T_emb.reshape(T_emb.shape[0], -1)
        #         info_S = info_abundance(S_emb)
        #         info_T = info_abundance(T_emb)
        #         print(info_S, info_T, end=" ")
        #         S_emb = self.projector(S_emb)
        #         T_emb = self.adaptor(T_emb)
        #         info_S = info_abundance(S_emb)
        #         info_T = info_abundance(T_emb)
        #         print(info_S, info_T)
        #         self.cnt += 1
        
        return loss


class PairD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "paird"
        self.beta = args.paird_beta
        self.tau = args.paird_tau

    def do_something_in_each_epoch(self, epoch):
        if epoch != 0: print(self.grad1, self.grad2)
        self.cnt = 0
        self.grad1, self.grad2 = 0., 0.

    def get_loss(self, feature, label):
        logit_S = self.student(feature).squeeze(1)
        logit_T = self.teacher(feature).squeeze(1)

        # randidx = torch.arange(len(logit_T)).flip(-1)
        randidx = torch.randperm(len(logit_T))
        neg_T, pos_T = logit_T.clone(), logit_T[randidx].clone()
        idx = torch.argwhere(neg_T > pos_T)
        neg_T[idx], pos_T[idx] = pos_T[idx], neg_T[idx]
        neg_S, pos_S = logit_S.clone(), logit_S[randidx].clone()
        neg_S[idx], pos_S[idx] = pos_S[idx], neg_S[idx]
        gap_T = pos_T - neg_T
        gap_S = pos_S.detach() - neg_S
        idx = torch.argwhere(neg_S < torch.quantile(logit_S, 0.01).detach())
        y_T = torch.sigmoid(gap_T[idx] / self.tau)
        y_S = torch.sigmoid(gap_S[idx] / self.tau)

        loss_rk = F.binary_cross_entropy(y_S, y_T)
        y_T = torch.sigmoid(logit_T)
        loss_bce = F.binary_cross_entropy_with_logits(logit_S, y_T)
        loss = self.beta * loss_rk + (1. - self.beta) * loss_bce
        
        with torch.no_grad():
            # grad1 = (torch.sigmoid(logit_S) - torch.sigmoid(logit_T)).mean().detach().cpu().item()
            # grad2 = (torch.sigmoid(gap_T) - torch.sigmoid(gap_S)).mean().detach().cpu().item()
            grad1 = torch.autograd.grad(loss_bce, logit_S, retain_graph=True)[0].sum().detach().cpu().item()
            grad2 = torch.autograd.grad(loss_rk, logit_S, retain_graph=True)[0].sum().detach().cpu().item()
            self.grad1 = (self.grad1 * self.cnt + grad1) / (self.cnt + 1)
            self.grad2 = (self.grad2 * self.cnt + grad2) / (self.cnt + 1)
        return loss


class FFFit(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "fffit"
        self.num_fields = self.student.num_fields
        self.projectors = nn.ModuleList([nn.Linear(self.student.embedding_dim, self.teacher.embedding_dim) for _ in range(self.num_fields)])
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_embed(data)    # bs, num_fields, embedding_dim
        T_emb = self.teacher.forward_embed(data)
        loss = 0.
        for field in range(self.num_fields):
            projector = self.projectors[field]
            T_field_emb = T_emb[:, field, :]
            S_field_emb = projector(S_emb[:, field, :])
            loss += (T_field_emb.detach() - S_field_emb).pow(2).sum(-1).mean()
        return loss


class AnyD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "anyd"
        self.T_layer = args.T_layer
        self.S_layer = args.S_layer
        self.T_dim = self.teacher.get_layer_dim(self.T_layer)
        self.S_dim = self.student.get_layer_dim(self.S_layer)
        self.projector = nn.Linear(self.S_dim, self.T_dim)
        nn.init.xavier_normal_(self.projector.weight)
        nn.init.constant_(self.projector.bias, 0)

    def get_loss(self, data, label):
        S_emb = self.student.forward_layer(data, self.S_layer)    # bs, layer_dim
        T_emb = self.teacher.forward_layer(data, self.T_layer)
        S_emb = self.projector(S_emb)
        loss = (T_emb.detach() - S_emb).pow(2).sum(-1).mean()
        return loss


class adaD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "adad"
        self.beta = args.adad_beta
        self.gamma = args.adad_gamma
        self.lmbda = args.lmbda
        self.verbose = args.verbose

        student_penultimate_dim = self.student._penultimate_dim
        if isinstance(self.teacher._penultimate_dim, int):
            # For one-stream models
            teacher_penultimate_dim = self.teacher._penultimate_dim
        else:
            # For two-stream models
            cross_dim, deep_dim = self.teacher._penultimate_dim
            teacher_penultimate_dim = cross_dim + deep_dim
        self.projector = nn.Linear(student_penultimate_dim, teacher_penultimate_dim)
        self.adaptor_S = nn.Linear(student_penultimate_dim, student_penultimate_dim)
        self.adaptor_T = nn.Linear(teacher_penultimate_dim, teacher_penultimate_dim)
        self.predictor_S = nn.Linear(student_penultimate_dim, 1)
        self.predictor_T = nn.Linear(teacher_penultimate_dim, 1)
    
    def do_something_in_each_epoch(self, epoch):
        # for embedding in self.student.embedding_layer.embedding.items():
        #     print(int(info_abundance(embedding[1].weight.data)), end=" ")
        # print()
        # for embedding in self.teacher.embedding_layer.embedding.items():
        #     print(int(info_abundance(embedding[1].weight.data)), end=" ")
        if self.verbose:
            if epoch > 0 and not self.cka is None:
                print(self.cka)
            self.cka = None
            self.cnt = 0
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        if isinstance(self.teacher._penultimate_dim, tuple):
            # Two-stream models
            T_emb_cross, T_emb_deep = T_emb
            T_emb = torch.cat([T_emb_cross, T_emb_deep], dim=-1)
        T_emb_adapt = self.adaptor_T(T_emb)
        S_emb_adapt = self.adaptor_S(S_emb.detach())
        T_logits_adapt = self.predictor_T(T_emb_adapt)
        S_logits_adapt = self.predictor_S(S_emb_adapt)
        T_pred = torch.sigmoid(self.teacher(data))
        S_pred = torch.sigmoid(self.student(data))
        loss_adapt_T = F.binary_cross_entropy_with_logits(T_logits_adapt.squeeze(-1), label.squeeze(-1).float()) + self.beta * F.binary_cross_entropy_with_logits(T_logits_adapt.squeeze(-1), S_pred.detach().squeeze(-1))
        loss_adapt_S = F.binary_cross_entropy_with_logits(S_logits_adapt.squeeze(-1), label.squeeze(-1).float()) + self.beta * F.binary_cross_entropy_with_logits(S_logits_adapt.squeeze(-1), T_pred.detach().squeeze(-1))
        adaptor_S_detach = deepcopy(self.adaptor_S)
        for param in adaptor_S_detach.parameters():
                param.requires_grad = False
        S_emb_proj = self.projector(adaptor_S_detach(S_emb))
        loss = (T_emb_adapt.detach() - S_emb_proj).pow(2).sum(-1).mean() + (loss_adapt_S + loss_adapt_T) / (self.lmbda + 1e-8) * self.gamma
        
        if self.verbose and self.cnt < 5:
            # # calculate CKA
            # with torch.no_grad():
            #     S_embs = self.student.forward_all_feature(data)
            #     T_embs = self.teacher.forward_all_feature(data)
            #     T_embs += [T_emb.detach()]
            #     CKA_mat = np.zeros((len(T_embs), len(S_embs)))
            #     for id_T, T_emb in enumerate(T_embs):
            #         for id_S, S_emb in enumerate(S_embs):
            #             CKA_mat[id_T, id_S] = CKA(T_emb, S_emb).item()
            #     if self.cka is None:
            #         self.cka = CKA_mat
            #     else:
            #         self.cka = (self.cka * self.cnt + CKA_mat) / (self.cnt + 1)
            #         self.cnt += 1

            # calculate information abundance
            with torch.no_grad():
                info_S = info_abundance(S_emb)
                info_T = info_abundance(T_emb)
                info_adapt_S = info_abundance(S_emb_adapt)
                info_adapt_T = info_abundance(T_emb_adapt)
                info_proj_S = info_abundance(S_emb_proj)
                print(info_S, info_T, info_adapt_S, info_adapt_T, info_proj_S)
                self.cnt += 1

            # calculate information abundance for each field
            # with torch.no_grad():
            #     # for i in range(self.student.num_fields):
            #     #     emb = S_emb[:, i*self.student.embedding_dim:(i+1)*self.student.embedding_dim]
            #     #     print(int(info_abundance(emb)), end=" ")
            #     # print()
            #     for i in range(self.teacher.num_fields):
            #         emb = T_emb[:, i*self.teacher.embedding_dim:(i+1)*self.teacher.embedding_dim]
            #         print(int(info_abundance(emb)), end=" ")
            #     print()
            #     self.cnt += 1
        
        return loss


class copyD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "copyd"
        self.num_experts = args.num_experts
        self.beta = args.copyd_beta
        self.gamma = args.copyd_gamma
        self.teacher_linear = self.teacher.linear
        self.projector = Projector(self.student._penultimate_dim, self.teacher._penultimate_dim, self.num_experts, norm=False, dropout_rate=0., shallow=False)
    
    # def get_ratings(self, data):
    #     old_linear = deepcopy(self.student.linear)
    #     self.student.linear = nn.Sequential(self.projector, self.teacher_linear)
    #     res = self.student(data)
    #     self.student.linear = old_linear
    #     return res
        
    def do_something_in_each_epoch(self, epoch):
        self.epoch = epoch
        if epoch != 0: print(f"loss_emb:{self.loss_emb}, loss_linear:{self.loss_linear}")
        self.loss_emb, self.loss_linear, self.cnt = 0, 0, 0
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        S_emb_proj = self.projector(S_emb)
        loss_emb = (T_emb.detach() - S_emb_proj).pow(2).sum(-1).mean()
        # S_logits = self.student.linear(S_emb.detach())
        # proj_linear_logits = self.teacher_linear(S_emb_proj).detach()
        # y_T = torch.sigmoid(proj_linear_logits)
        # loss_linear = F.binary_cross_entropy_with_logits(S_logits, y_T.float())
        # loss_linear = F.binary_cross_entropy_with_logits(S_logits.squeeze(-1), label.squeeze(-1).float())
        # for _, module in self.student.linear.named_modules():
        #         for p_name, param in module.named_parameters():
        #             if param.requires_grad:
        #                 if p_name in ["weight", "bias"]:
        #                     loss_linear += torch.norm(param, p=2) * self.student.L2_weight
        loss = loss_emb
        self.loss_emb = (self.loss_emb * self.cnt + loss_emb.detach().item()) / (self.cnt + 1)
        # self.loss_linear = (self.loss_linear * self.cnt + loss_linear.detach().item()) / (self.cnt + 1)
        self.cnt += 1
        return loss

    def forward(self, data, label):
        # self.student.linear = nn.Sequential(self.projector, self.teacher_linear)
        output = self.student(data)
        base_loss = self.student.get_loss(output, label)
        kd_loss = self.get_loss(data, label)
        loss = kd_loss + self.lmbda * base_loss
        return loss, base_loss.detach(), kd_loss.detach()


class fieldD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "fieldd"
        self.num_experts = args.num_experts
        self.beta = args.fieldd_beta
        self.gamma = args.fieldd_gamma
        self.teacher_linear = self.teacher.linear
        self.projectors = nn.ModuleList([Projector(self.student._penultimate_dim, self.teacher._penultimate_dim // self.student.num_fields, 1, norm=False, dropout_rate=0., shallow=True) for _ in range(self.student.num_fields)])

    # def get_ratings(self, data):
    #     old_linear = deepcopy(self.student.linear)
    #     self.student.linear = nn.Sequential(self.projector, self.teacher_linear)
    #     res = self.student(data)
    #     self.student.linear = old_linear
    #     return res
    
    def do_something_in_each_epoch(self, epoch):
        self.epoch = epoch
        if epoch != 0: print(f"loss_emb:{self.loss_emb}, loss_linear:{self.loss_linear}")
        self.loss_emb, self.loss_linear, self.cnt = 0, 0, 0
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        S_emb_proj = []
        for i in range(self.teacher.num_fields):
            S_emb_proj.append(self.projectors[i](S_emb))
        S_emb_proj = torch.cat(S_emb_proj, dim=-1)
        loss_emb = (T_emb.detach() - S_emb_proj).pow(2).sum(-1).mean()
        S_logits = self.student.linear(S_emb.detach())
        proj_linear_logits = self.teacher_linear(S_emb_proj).detach()
        y_T = torch.sigmoid(proj_linear_logits)
        loss_linear = F.binary_cross_entropy_with_logits(S_logits, y_T.float())
        loss_linear = self.gamma * loss_linear + (1. - self.gamma) * F.binary_cross_entropy_with_logits(S_logits.squeeze(-1), label.squeeze(-1).float())
        loss = self.beta * loss_emb + (1. - self.beta) * loss_linear
        self.loss_emb = (self.loss_emb * self.cnt + loss_emb.detach().item()) / (self.cnt + 1)
        self.loss_linear = (self.loss_linear * self.cnt + loss_linear.detach().item()) / (self.cnt + 1)
        self.cnt += 1
        return loss

    def forward(self, data, label):
        output = self.student(data)
        base_loss = self.student.get_loss(output, label)
        kd_loss = self.get_loss(data, label)
        loss = kd_loss
        return loss, base_loss.detach(), kd_loss.detach()


class watD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "watd"
        self.beta = args.watd_beta
        self.adaptor = Projector(self.teacher._penultimate_dim, self.student._penultimate_dim, 1, norm=False, dropout_rate=0., shallow=True)
        self.predictor = nn.Linear(self.student._penultimate_dim, 1)
    
    def do_something_in_each_epoch(self, epoch):
        self.epoch = epoch
        if epoch != 0: print(f"loss_emb:{self.loss_emb}, loss_adapt:{self.loss_adapt}")
        self.loss_emb, self.loss_adapt, self.cnt = 0, 0, 0
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data).detach()
        T_emb_adapt = self.adaptor(T_emb)
        T_logit = self.teacher(data)
        T_adapt_logit = self.predictor(T_emb_adapt)
        loss_emb = (T_emb_adapt.detach() - S_emb).pow(2).sum(-1).mean()
        T_pred = torch.sigmoid(T_logit)
        loss_adapt = F.binary_cross_entropy_with_logits(T_adapt_logit, T_pred)
        loss = loss_emb + self.beta * loss_adapt
        self.loss_emb = (self.loss_emb * self.cnt + loss_emb.detach().item()) / (self.cnt + 1)
        self.loss_adapt = (self.loss_adapt * self.cnt + loss_adapt.detach().item()) / (self.cnt + 1)
        self.cnt += 1
        return loss

    def forward(self, data, label):
        output = self.student(data)
        base_loss = self.student.get_loss(output, label)
        kd_loss = self.get_loss(data, label)
        loss = kd_loss
        return loss, base_loss.detach(), kd_loss.detach()


class attachD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "attachd"
        self.num_experts = args.num_experts
        self.teacher_linear = self.teacher.linear
        self.projector = Projector(self.student._penultimate_dim, self.teacher._penultimate_dim, self.num_experts, norm=False, dropout_rate=0., shallow=False)
        self.student.linear = Projector(self.student._penultimate_dim, 1, 1, norm=False, dropout_rate=0., shallow=False)
    
    # def get_ratings(self, data):
    #     old_linear = deepcopy(self.student.linear)
    #     self.student.linear = nn.Sequential(self.projector, self.teacher_linear)
    #     res = self.student(data)
    #     self.student.linear = old_linear
    #     return res
        
    def do_something_in_each_epoch(self, epoch):
        self.epoch = epoch
        if epoch != 0: print(f"loss_emb:{self.loss_emb}, loss_linear:{self.loss_linear}")
        self.loss_emb, self.loss_linear, self.cnt = 0, 0, 0
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        S_emb_proj = self.projector(S_emb)
        loss_emb = (T_emb.detach() - S_emb_proj).pow(2).sum(-1).mean()
        S_logits = self.student.linear(S_emb.detach())
        proj_linear_logits = self.teacher_linear(S_emb_proj).detach()
        y_T = torch.sigmoid(proj_linear_logits)
        loss_linear = F.binary_cross_entropy_with_logits(S_logits, y_T.float())
        loss = loss_emb + loss_linear
        self.loss_emb = (self.loss_emb * self.cnt + loss_emb.detach().item()) / (self.cnt + 1)
        self.loss_linear = (self.loss_linear * self.cnt + loss_linear.detach().item()) / (self.cnt + 1)
        self.cnt += 1
        return loss

    def forward(self, data, label):
        output = self.student(data)
        base_loss = self.student.get_loss(output, label)
        kd_loss = self.get_loss(data, label)
        loss = kd_loss
        return loss, base_loss.detach(), kd_loss.detach()


class TCAD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "tcad"
        self.layer = args.tcad_layer
        self.lmbda = args.tcad_lmbda
        self.K = args.tcad_K
        self.verbose = args.verbose
        self.ablation = args.ablation
        if self.layer == "embedding":
            self.tca = TCA(self.student.embedding_layer_dim, self.lmbda, self.K)
            self.projector = Projector(self.student.embedding_layer_dim, self.teacher.embedding_layer_dim, 1, norm=False, dropout_rate=0., shallow=True)
        elif self.layer == "penultimate":
            self.tca = TCA(self.student._penultimate_dim, self.lmbda, self.K, False)
            self.projector = Projector(self.student._penultimate_dim, self.teacher._penultimate_dim, 1, norm=False, dropout_rate=0., shallow=True)
        elif self.layer == "none":
            pass
        else:
            raise ValueError
        self.step = 0
    
    def get_loss(self, data, label):
        if self.layer == "embedding":
            S_emb = self.student.forward_embed(data)
            S_emb = S_emb.reshape(S_emb.shape[0], -1)
            T_emb = self.teacher.forward_embed(data)
            T_emb = T_emb.reshape(T_emb.shape[0], -1)
        elif self.layer == "penultimate":
            S_emb = self.student.forward_penultimate(data)
            T_emb = self.teacher.forward_penultimate(data)
        if self.layer in ["embedding", "penultimate"]:
            if self.ablation:
                S_emb_adapt = S_emb / torch.linalg.norm(S_emb, dim=-1, keepdim=True).detach()
                T_emb_adapt = T_emb / torch.linalg.norm(T_emb, dim=-1, keepdim=True).detach()
            else:
                if self.step % 500 == 0:
                    # S_emb_pos, S_emb_neg = S_emb[label == 1], S_emb[label == 0]
                    # T_emb_pos, T_emb_neg = T_emb[label == 1], T_emb[label == 0]
                    # S_emb_pos_adapt, T_emb_pos_adapt = self.tca.fit_transform(S_emb_pos, T_emb_pos)
                    # S_emb_neg_adapt, T_emb_neg_adapt = self.tca.fit_transform(S_emb_neg, T_emb_neg)
                    S_emb_adapt, T_emb_adapt = self.tca.fit_transform(S_emb, T_emb, label)
                    loss = (T_emb_adapt.detach() - S_emb_adapt).pow(2).sum(-1).mean()
                    mmd = MMD_loss()(S_emb_adapt[-512:], T_emb_adapt[-512:])
                    # mmd_pos = MMD_loss()(S_emb_pos_adapt[-512:], T_emb_pos_adapt[-512:])
                    # mmd_neg = MMD_loss()(S_emb_neg_adapt[-512:], T_emb_neg_adapt[-512:])
                    if self.verbose:
                        mlflow.log_metrics({"mmd":mmd.item()}, step=self.step)
                        mlflow.log_metrics({"info_S_emb":info_abundance(S_emb), "info_T_emb":info_abundance(T_emb), "info_S_adapt":info_abundance(S_emb_adapt), "info_T_adapt":info_abundance(T_emb_adapt)}, step=self.step)
                        # mlflow.log_metrics({"info_S_emb_pos":info_abundance(S_emb_pos), "info_T_emb_pos":info_abundance(T_emb_pos), "info_S_pos_adapt":info_abundance(S_emb_pos_adapt), "info_T_pos_adapt":info_abundance(T_emb_pos_adapt)}, step=self.step)
                        # mlflow.log_metrics({"info_S_emb_neg":info_abundance(S_emb_neg), "info_T_emb_neg":info_abundance(T_emb_neg), "info_S_neg_adapt":info_abundance(S_emb_neg_adapt), "info_T_neg_adapt":info_abundance(T_emb_neg_adapt)}, step=self.step)
                else:
                    loss = torch.tensor(0.).cuda()
                self.step += 1
        elif self.layer == "none":
            loss = torch.tensor(0.).cuda()
        else: raise ValueError
        return loss


class DAND(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "dand"
        self.layer = args.dand_layer
        self.K = args.dand_K
        self.kernel_type = args.kernel_type
        self.ablation = args.ablation
        if self.layer == "embedding":
            self.mmd = MMD_loss(self.kernel_type)
            self.projector = Projector(self.student.embedding_layer_dim, self.teacher.embedding_layer_dim, 1, norm=False, dropout_rate=0., shallow=True)
        elif self.layer == "penultimate":
            self.mmd = MMD_loss(self.kernel_type)
            self.projector = Projector(self.student._penultimate_dim, self.teacher._penultimate_dim, 1, norm=False, dropout_rate=0., shallow=True)
        elif self.layer == "none":
            pass
        else:
            raise ValueError
    
    def get_loss(self, data, label):
        if self.layer == "embedding":
            S_emb = self.student.forward_embed(data)
            S_emb = S_emb.reshape(S_emb.shape[0], -1)
            T_emb = self.teacher.forward_embed(data)
            T_emb = T_emb.reshape(T_emb.shape[0], -1)
        elif self.layer == "penultimate":
            S_emb = self.student.forward_penultimate(data)
            T_emb = self.teacher.forward_penultimate(data)
        if self.layer in ["embedding", "penultimate"]:
            S_emb_proj = self.projector(S_emb)
            if self.ablation:
                loss = (T_emb - S_emb_proj).pow(2).sum(-1).mean()
            else:
                if self.kernel_type == "linear":
                    loss = self.mmd(S_emb_proj, T_emb)
                else:
                    loss = self.mmd(S_emb_proj[:self.K], T_emb[:self.K])
        elif self.layer == "none":
            loss = torch.tensor(0.).cuda()
        else: raise ValueError
        return loss


class MixD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "mixd"
        self.layer = args.mixd_layer
        self.K = args.mixd_K
        self.kernel_type = args.kernel_type
        self.verbose = args.verbose
        self.ablation = args.ablation
        if self.layer == "embedding":
            self.mmd = MMD_loss(self.kernel_type)
            self.projector_pos = Projector(self.student.embedding_layer_dim, self.teacher.embedding_layer_dim, 1, norm=False, dropout_rate=0., shallow=True)
            self.projector_neg = Projector(self.student.embedding_layer_dim, self.teacher.embedding_layer_dim, 1, norm=False, dropout_rate=0., shallow=True)
        elif self.layer == "penultimate":
            self.mmd = MMD_loss(self.kernel_type)
            self.projector_pos = Projector(self.student._penultimate_dim, self.teacher._penultimate_dim, 1, norm=False, dropout_rate=0., shallow=True)
            self.projector_neg = Projector(self.student._penultimate_dim, self.teacher._penultimate_dim, 1, norm=False, dropout_rate=0., shallow=True)
        elif self.layer == "none":
            pass
        else:
            raise ValueError
    
    def _cal_loss(self, S_emb, T_emb, projector):
        S_emb_proj = projector(S_emb)
        if self.ablation:
            loss = (T_emb[:self.K] - S_emb_proj[:self.K]).pow(2).sum(-1).mean()
            return loss
        if self.kernel_type == "linear":
            loss = self.mmd(S_emb_proj[:self.K], T_emb[:self.K])
        else:
            loss = self.mmd(S_emb_proj[:self.K], T_emb[:self.K])
        return loss
    
    def get_loss(self, data, label):
        if self.layer == "embedding":
            S_emb = self.student.forward_embed(data)
            S_emb = S_emb.reshape(S_emb.shape[0], -1)
            T_emb = self.teacher.forward_embed(data)
            T_emb = T_emb.reshape(T_emb.shape[0], -1)
        elif self.layer == "penultimate":
            S_emb = self.student.forward_penultimate(data)
            T_emb = self.teacher.forward_penultimate(data)
        if self.layer in ["embedding", "penultimate"]:
            idx_pos = torch.argwhere(label == 1).squeeze(-1)
            idx_neg = torch.argwhere(label == 0).squeeze(-1)
            S_emb_pos, S_emb_neg = S_emb[idx_pos], S_emb[idx_neg]
            T_emb_pos, T_emb_neg = T_emb[idx_pos], T_emb[idx_neg]
            loss_pos = self._cal_loss(S_emb_pos, T_emb_pos, self.projector_pos)
            loss_neg = self._cal_loss(S_emb_neg, T_emb_neg, self.projector_neg)
            loss = (loss_pos + loss_neg) * 0.5
            if self.verbose:
                mlflow.log_metrics({"loss_pos":loss_pos.item(), "loss_neg":loss_neg.item()})
        elif self.layer == "none":
            loss = torch.tensor(0.).cuda()
        else: raise ValueError
        return loss


class RSD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "rsd"
        self.ratio = args.rsd_ratio
    
    def cal_rsd(self, X, Y):
        Us, Ss, Vs = torch.svd(X)
        Ut, St, Vt = torch.svd(Y)
        total_Ss, total_St = torch.sum(Ss), torch.sum(St)
        cumu_Ss, cumu_St = torch.cumsum(Ss, dim=0), torch.cumsum(St, dim=0)
        thresh_s, thresh_t = self.ratio * total_Ss, self.ratio * total_St
        K_s, K_t = torch.where(cumu_Ss >= thresh_s)[0][0] + 1, torch.where(cumu_St >= thresh_t)[0][0] + 1
        Us, Ut = Us[:, :K_s], Ut[:, :K_t]
        Ps, cospa, Pt = torch.svd(torch.mm(Us.T, Ut))
        cospa = torch.minimum(cospa, torch.tensor(1.))
        sinpa = torch.sqrt(1. - torch.pow(cospa, 2))
        rsd = torch.norm(sinpa, 1)
        # bmp = torch.norm(Ps.abs() - Pt.abs(), 2)
        dis = rsd
        return dis
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        loss = self.cal_rsd(S_emb, T_emb)
        return loss


class PCAD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "pcad"
        self.SK = args.pcad_SK
        self.TK = args.pcad_TK
        self.Sr = args.pcad_Sr
        self.Tr = args.pcad_Tr
        self.projector = Projector(self.SK, self.TK, 1, norm=False, dropout_rate=0., shallow=True)

    def pca_with_grad(self, X, K, rat):
        # Xc = X - X.mean(0, keepdims=True)
        U, S, V = torch.svd_lowrank(X, K)
        S_smooth = S.clone()
        S_smooth[:math.ceil(K * rat)] = S[:math.ceil(K * rat)].mean()
        Y = U.mm(torch.diag(S_smooth))
        return Y
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        S_emb = self.pca_with_grad(S_emb, self.SK, self.Sr)
        T_emb = self.pca_with_grad(T_emb, self.TK, self.Tr)
        S_emb_proj = self.projector(S_emb)
        loss = (T_emb - S_emb_proj).pow(2).sum(-1).mean()
        return loss


class CCD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "ccd"
        self.SK = args.ccd_SK
        self.TK = args.ccd_TK
        self.beta = args.ccd_beta
        self.projector = Projector(self.SK, self.TK, 1, norm=False, dropout_rate=0., shallow=True)
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        S_emb = S_emb - S_emb.mean(0, keepdims=True)
        T_emb = T_emb - T_emb.mean(0, keepdims=True)
        Us, Ss, Vs = torch.svd_lowrank(S_emb, self.SK)
        Ut, St, Vt = torch.svd_lowrank(T_emb, self.TK)
        # Us, Ss, Vs = torch.svd(S_emb)
        # Ut, St, Vt = torch.svd(T_emb)
        # Us, Ss, Vs = Us[:, :self.SK], Ss[:self.SK], Vs[:, :self.SK]
        # Ut, St, Vt = Ut[:, :self.TK], St[:self.TK], Vt[:, :self.TK]
        
        # Ys = Us.mm(torch.diag(Ss))     # bs, SK
        # Yt = Ut.mm(torch.diag(St))     # bs, TK
        # loss1 = self.projector(Ys.detach()).pow(2).sum() + Yt.pow(2).sum()
        # # mlflow.log_metric("rk_proj", info_abundance(self.projector.experts[0].mlp[0].weight.data))
        # vec_t = Ut                  # bs, TK
        # # Ss_smooth = Ss.detach().clone()
        # # Ss_smooth[:math.ceil(self.SK * 0.3)] = Ss[:math.ceil(self.SK * 0.3)].mean()
        # # Us = Us * Ss_smooth.unsqueeze(0)   # bs, SK
        # vec_s = self.projector(Us)  # bs, TK

        # vec_t = vec_t / torch.norm(vec_t, dim=0, keepdim=True)
        # vec_s = vec_s / torch.norm(vec_s, dim=0, keepdim=True)
        # cos_mat = torch.diag(vec_t.T.mm(vec_s))     # TK,
        # sin_mat = torch.sqrt(1. - torch.pow(torch.minimum(cos_mat, torch.tensor(1.)), 2))
        # # St_smooth = St.clone()
        # # St_smooth[:math.ceil(self.TK * 0.2)] = St[:math.ceil(self.TK * 0.2)].mean()

        # # d = {f"sin_{i}":sin_mat[i].item() for i in range(10)}
        # # mlflow.log_metrics(d)

        # # St[10:] = 0
        # # St[[1, 2, 6, 7, 8, 9]] = 0
        # # St[1:] = 0

        idx = [0]
        cos_mat = torch.diag(Ut[:, idx].T.mm(Us[:, idx]))
        sin_mat = torch.sqrt(1. - torch.pow(torch.minimum(cos_mat, torch.tensor(1.)), 2))
        mlflow.log_metric("sin_00", sin_mat.mean().item())
        mlflow.log_metric("St", St[idx].mean().item())
        mlflow.log_metric("Ss", Ss[idx].mean().item())
        sin_mat = sin_mat * St[idx] * 10

        # sin_mat = sin_mat * St
        loss2 = 2 * sin_mat.sum()
        # loss = (loss1 * self.beta + loss2) / len(S_emb)
        loss = loss2 / len(S_emb)
        # mlflow.log_metric("loss2", loss2.item())

        # P = self.projector.experts[0].mlp[0].weight.T[:, :10]
        # lam = torch.linalg.svdvals(P)
        # loss3 = -lam.pow(2).sum()
        # loss = (loss1 * self.beta + loss2 + loss3) / len(S_emb)
        # mlflow.log_metric("rk_loss", loss3.item())
        return loss


class conD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "cond"
        self.beta = args.cond_beta
        self.adaptor = Projector(self.teacher._penultimate_dim, self.student._penultimate_dim, 1, norm=False, dropout_rate=0., shallow=False)
        self.projector = Projector(self.student._penultimate_dim, self.student._penultimate_dim, 1, norm=False, dropout_rate=0., shallow=True)

    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        T_adapt_emb = self.adaptor(T_emb)
        predictor = deepcopy(self.student.linear)
        for param in predictor.parameters():
            param.requires_grad = False
        T_logits = predictor(T_adapt_emb)
        loss_T = F.binary_cross_entropy_with_logits(T_logits.squeeze(-1), label.squeeze(-1).float())
        T_adapt_proj = T_adapt_emb
        loss_S = (T_adapt_proj.detach() - S_emb).pow(2).sum(-1).mean()
        loss = loss_T * self.beta + loss_S
        mlflow.log_metrics({"loss_T":loss_T.item(), "loss_S":loss_S.item()})
        return loss


class PDD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "pdd"
        self.K = args.pdd_K
        self.beta = args.pdd_beta
        self.gamma = args.pdd_gamma
        self.adaptor = Projector(self.teacher._penultimate_dim, self.student._penultimate_dim, 1, norm=False, dropout_rate=0.1, shallow=False)
    
    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)
        T_emb = self.teacher.forward_penultimate(data)
        T_emb_adapt = self.adaptor(T_emb)
        predictor = deepcopy(self.student.linear)
        for param in predictor.parameters():
            param.requires_grad = False
        T_logits = predictor(T_emb_adapt)
        loss_T = F.binary_cross_entropy_with_logits(T_logits.squeeze(-1), label.squeeze(-1).float())

        Us, Ss, Vs = torch.svd_lowrank(S_emb, self.K)
        Ut, St, Vt = torch.svd_lowrank(T_emb_adapt.detach(), self.K)
        loss1 = Ss.sum()
        cos = torch.diag(Ut.T.mm(Us))
        sin = torch.sqrt(1. - torch.pow(torch.minimum(cos, torch.tensor(1.)), 2))
        loss2 = sin.sum()
        # loss2 = -cos.sum()
        loss = loss_T * self.gamma + loss1 + self.beta * loss2
        return loss


class shrD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "shrd"
        self.K = args.shrd_K
        self.beta = args.shrd_beta
        self.layer = args.shrd_layer
        
    def get_loss(self, data, label):
        if self.layer == "penu":
            S_emb = self.student.forward_penultimate(data)
            T_emb = self.teacher.forward_penultimate(data)
        elif self.layer == "embed":
            S_emb = self.student.forward_embed(data)
            S_emb = S_emb.reshape(S_emb.shape[0], -1)
            T_emb = self.teacher.forward_embed(data)
            T_emb = T_emb.reshape(T_emb.shape[0], -1)
        else: raise ValueError
        S_emb = S_emb - S_emb.mean(0, keepdims=True)
        T_emb = T_emb - T_emb.mean(0, keepdims=True)
        Us, Ss, Vs = torch.svd_lowrank(S_emb, self.K)
        Ut, St, Vt = torch.svd_lowrank(T_emb, self.K)
        # Pt = torch.softmax(St / self.T, dim=0)
        # loss1 = F.cross_entropy(Ss, Pt)
        loss1 = Ss.sum()
        cos = torch.diag(Ut.T.mm(Us))
        sin = torch.sqrt(1. - torch.pow(torch.minimum(cos, torch.tensor(1.)), 2))
        loss2 = sin.sum()
        mlflow.log_metrics({f"loss2_{i}": sin[i] for i in range(self.K)})
        # loss2 = -cos.sum()
        loss = loss2 + self.beta * loss1
        mlflow.log_metrics({"loss1":loss1.item(), "loss2":loss2.item()})
        return loss
    
    # def forward(self, data, label):
    #     output = self.student(data)
    #     base_loss = self.student.get_loss(output, label)
    #     kd_loss = self.get_loss(data, label)
    #     loss = self.lmbda * kd_loss
    #     return loss, base_loss.detach(), kd_loss.detach()