import os
import re
import math
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from .utils import Expert
from .base_model import BaseKD4Rec


class Scratch(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()

        self.args = args
        self.backbone = backbone
        self.training = True

    def get_ratings(self, param):
        if self.args.task == "ctr":
            return self.backbone(param)
        else:
            return self.backbone.get_ratings(param)

    def do_something_in_each_epoch(self, epoch):
        return
    
    def train(self):
        self.training = True
        self.backbone.train()

    def eval(self):
        self.training = False
        self.backbone.eval()
    
    def get_params_to_update(self):
        return [{"params": self.backbone.parameters(), 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def forward(self, *params):
        if self.args.task == "ctr":
            data, labels = params
            output = self.backbone(data)
            base_loss = self.backbone.get_loss(output, labels)
        else:
            output = self.backbone(*params)
            base_loss = self.backbone.get_loss(output)
        loss = base_loss
        return loss

    @property
    def param_to_save(self):
        return self.backbone.state_dict()


class RD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)

        self.mu = args.rd_mu
        self.topk = args.rd_topk
        self.T = args.rd_T
        self.dynamic_sample_num = args.rd_dynamic_sample
        self.start_epoch = args.rd_start_epoch
        
        self.rank_aware = False
        self.RANK = None
        self.epoch = 0
        self._weight_renormalize = True

        self._generateTopK()
        self._static_weights = self._generateStaticWeights()

    def Sample_neg(self, dns_k):
        """python implementation for 'UniformSample_DNS'
        """
        S = []
        BinForUser = np.zeros(shape=(self.num_items, )).astype("int")
        for user in range(self.num_users):
            posForUser = list(self.dataset.train_dict[user])
            if len(posForUser) == 0:
                continue
            BinForUser[:] = 0
            BinForUser[posForUser] = 1
            NEGforUser = np.where(BinForUser == 0)[0]
            negindex = np.random.randint(0, len(NEGforUser), size=(dns_k, ))
            negitems = NEGforUser[negindex]
            add_pair = [*negitems]
            S.append(add_pair)
        return S

    def do_something_in_each_epoch(self, epoch):
        self.epoch = epoch
        self.dynamic_samples = self.Sample_neg(self.dynamic_sample_num)
        self.dynamic_samples = torch.Tensor(self.dynamic_samples).long().cuda()

    def _generateStaticWeights(self):
        w = torch.arange(1, self.topk + 1).float()
        w = torch.exp(-w / self.T)
        return (w / w.sum()).unsqueeze(0)

    def _generateTopK(self):
        if self.RANK is None:
            with torch.no_grad():
                self.RANK = torch.zeros((self.num_users, self.topk)).cuda()
                scores = self.teacher.get_all_ratings()
                train_pairs = self.dataset.train_pairs
                scores[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
                self.RANK = torch.topk(scores, self.topk)[1]

    def _weights(self, S_score_in_T, epoch, dynamic_scores):
        batch = S_score_in_T.shape[0]
        if epoch < self.start_epoch:
            return self._static_weights.repeat((batch, 1)).cuda()
        with torch.no_grad():
            static_weights = self._static_weights.repeat((batch, 1))
            # ---
            topk = S_score_in_T.shape[-1]
            num_dynamic = dynamic_scores.shape[-1]
            m_items = self.num_items
            dynamic_weights = torch.zeros(batch, topk)
            for col in range(topk):
                col_prediction = S_score_in_T[:, col].unsqueeze(1)
                num_smaller = torch.sum(col_prediction < dynamic_scores, dim=1).float()
                # print(num_smaller.shape)
                relative_rank = num_smaller / num_dynamic
                appro_rank = torch.floor((m_items - 1) * relative_rank) + 1

                dynamic = torch.tanh(self.mu * (appro_rank - col))
                dynamic = torch.clamp(dynamic, min=0.)

                dynamic_weights[:, col] = dynamic.squeeze()
            if self._weight_renormalize:
                return F.normalize(static_weights * dynamic_weights,
                                   p=1,
                                   dim=1).cuda()
            else:
                return (static_weights * dynamic_weights).cuda()

    def get_loss(self, batch_users, batch_pos_item, batch_neg_item):
        dynamic_samples = self.dynamic_samples[batch_users]
        dynamic_scores = self.student.forward_multi_items(batch_users, dynamic_samples).detach()
        topk_teacher = self.RANK[batch_users]

        S_score_in_T = self.student.forward_multi_items(batch_users, topk_teacher)
        weights = self._weights(S_score_in_T.detach(), self.epoch, dynamic_scores)
        
        RD_loss = -(weights * torch.log(torch.sigmoid(S_score_in_T)))
        
        RD_loss = RD_loss.sum(1)
        RD_loss = RD_loss.sum()

        return  RD_loss
    

class CD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)

        self.sample_num = args.cd_sample_num
        self.strategy = args.cd_strategy
        self.T = args.cd_T
        self.n_distill = args.cd_n_distill
        self.t1 = args.cd_t1
        self.t2 = args.cd_t2
        
        ranking_list = torch.exp(-torch.arange(1, self.sample_num + 1).float() / self.sample_num / self.T)
        self.ranking_mat = torch.stack([ranking_list] * self.num_users, 0)
        self.ranking_mat.requires_grad = False
        if self.strategy == "random":
            self.MODEL = None
        elif self.strategy == "student guide":
            self.MODEL = self.student
        elif self.strategy == "teacher guide":
            self.MODEL = self.teacher
        else:
            raise TypeError("CD support [random, student guide, teacher guide], " \
                            f"But got {self.strategy}")
        self.get_rank_sample(self.MODEL)
    
    def do_something_in_each_epoch(self, epoch):
        if self.strategy == "student guide":
            self.get_rank_sample(self.MODEL)

    def random_sample(self, batch_size):
        samples = np.random.choice(self.num_items, (batch_size, self.n_distill))
        return torch.from_numpy(samples).long().cuda()

    def get_rank_sample(self, MODEL):
        if MODEL is None:
            self.rank_samples =  self.random_sample(self.num_users)
            return
        self.rank_samples = torch.zeros(self.num_users, self.n_distill)
        with torch.no_grad():
            scores = MODEL.get_all_ratings()
            train_pairs = self.dataset.train_pairs
            scores[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            rank_scores, rank_items = torch.topk(scores, self.sample_num)

            for user in range(self.num_users):
                ranking_list = self.ranking_mat[user]
                rating = rank_scores[user]
                negitems = rank_items[user]
                sampled_items = set()
                while True:
                    samples = torch.multinomial(ranking_list, 2, replacement=True)
                    if rating[samples[0]] > rating[samples[1]]:
                        sampled_items.add(negitems[samples[0]])
                    else:
                        sampled_items.add(negitems[samples[1]])
                    if len(sampled_items) >= self.n_distill:
                        break
                self.rank_samples[user] = torch.Tensor(list(sampled_items))
        self.rank_samples = self.rank_samples.cuda().long()


    def get_loss(self, batch_users, batch_pos_item, batch_neg_item):
        random_samples = self.rank_samples[batch_users, :]
        samples_scores_T = self.teacher.forward_multi_items(batch_users, random_samples)
        samples_scores_S = self.student.forward_multi_items(batch_users, random_samples)
        weights = torch.sigmoid((samples_scores_T + self.t2) / self.t1)
        inner = torch.sigmoid(samples_scores_S)
        CD_loss = -(weights * torch.log(inner + 1e-10) + (1 - weights) * torch.log(1 - inner + 1e-10))

        CD_loss = CD_loss.sum(1).sum()
        return CD_loss


class DE(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)

        self.max_epoch = args.epochs
        self.end_T = args.de_end_T
        self.anneal_size = args.de_anneal_size
        self.num_experts = args.de_num_experts
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.current_T = self.end_T * self.anneal_size

        expert_dims = [self.student_dim, (self.teacher_dim + self.student_dim) // 2, self.teacher_dim]
        self.user_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])
        self.item_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])

        self.user_selection_net = nn.Sequential(nn.Linear(self.teacher_dim, self.num_experts), nn.Softmax(dim=1))
        self.item_selection_net = nn.Sequential(nn.Linear(self.teacher_dim, self.num_experts), nn.Softmax(dim=1))

        self.sm = nn.Softmax(dim=1)

    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def do_something_in_each_epoch(self, epoch):
        self.current_T = self.end_T * self.anneal_size * ((1. / self.anneal_size) ** (epoch / self.max_epoch))
        self.current_T = max(self.current_T, self.end_T)


    def get_DE_loss(self, batch_entity, is_user=True):
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

        expert_outputs = [experts[i](s).unsqueeze(-1) for i in range(self.num_experts)] 		# s -> t
        expert_outputs = torch.cat(expert_outputs, -1)							# batch_size x teacher_dims x num_experts

        expert_outputs = expert_outputs * selection_result						# batch_size x teacher_dims x num_experts
        expert_outputs = expert_outputs.sum(2)								# batch_size x teacher_dims	

        DE_loss = ((t - expert_outputs) ** 2).sum(-1).sum()

        return DE_loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        DE_loss_user = self.get_DE_loss(batch_user.unique(), is_user=True)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), is_user=False)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), is_user=False)
        DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5
        return DE_loss


class RRD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        
        self.K = args.rrd_K
        self.L = args.rrd_L
        self.T = args.rrd_T
        self.mxK = args.rrd_mxK

        # For interesting item
        self.get_topk_dict()
        ranking_list = torch.exp(-(torch.arange(self.mxK) + 1) / self.T)
        self.ranking_mat = ranking_list.repeat(self.num_users, 1)

        # For uninteresting item
        self.mask = torch.ones((self.num_users, self.num_items))
        train_pairs = self.dataset.train_pairs
        self.mask[train_pairs[:, 0], train_pairs[:, 1]] = 0
        for user in range(self.num_users):
            self.mask[user, self.topk_dict[user]] = 0
        self.mask.requires_grad = False

    def get_topk_dict(self):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings()
            train_pairs = self.dataset.train_pairs
            # remove true interactions from topk_dict
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            _, self.topk_dict = torch.topk(inter_mat, self.mxK, dim=-1)
    
    def get_samples(self, batch_user):

        interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
        uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)

        return interesting_samples, uninteresting_samples

    # epoch 마다
    def do_something_in_each_epoch(self, epoch):
        with torch.no_grad():
            # interesting items
            self.interesting_items = torch.zeros((self.num_users, self.K))

            # sampling
            while True:
                samples = torch.multinomial(self.ranking_mat, self.K, replacement=False)
                if (samples > self.mxK).sum() == 0:
                    break

            samples = samples.sort(dim=1)[0]

            for user in range(self.num_users):
                self.interesting_items[user] = self.topk_dict[user][samples[user]]

            self.interesting_items = self.interesting_items.cuda()

            # uninteresting items
            m1 = self.mask[: self.num_users // 2, :].cuda()
            tmp1 = torch.multinomial(m1, self.L, replacement=False)
            del m1

            m2 = self.mask[self.num_users // 2 : ,:].cuda()
            tmp2 = torch.multinomial(m2, self.L, replacement=False)
            del m2

            self.uninteresting_items = torch.cat([tmp1, tmp2], 0)
    
    def relaxed_ranking_loss(self, S1, S2):
        
        S1 = torch.minimum(S1, torch.tensor(80., device=S1.device))     # This may help
        S2 = torch.minimum(S2, torch.tensor(80., device=S2.device))

        above = S1.sum(1, keepdims=True)

        below1 = S1.flip(-1).exp().cumsum(1)    # exp() of interesting_prediction results in inf
        below2 = S2.exp().sum(1, keepdims=True)

        below = (below1 + below2).log().sum(1, keepdims=True)
        
        return -(above - below).sum()

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        users = batch_user.unique()
        interesting_items, uninteresting_items = self.get_samples(users)
        interesting_items = interesting_items.type(torch.LongTensor).cuda()
        uninteresting_items = uninteresting_items.type(torch.LongTensor).cuda()

        interesting_prediction = self.student.forward_multi_items(users, interesting_items)
        uninteresting_prediction = self.student.forward_multi_items(users, uninteresting_items)

        URRD_loss = self.relaxed_ranking_loss(interesting_prediction, uninteresting_prediction)

        return URRD_loss


class HTD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)

        self.max_epoch = args.epochs
        self.alpha = args.htd_alpha
        self.num_experts = args.htd_num_experts
        self.choice = args.htd_choice
        self.T = args.htd_T

        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        # Group Assignment related parameters
        F_dims = [self.student_dim, (self.teacher_dim + self.student_dim) // 2, self.teacher_dim]

        self.user_f = nn.ModuleList([Expert(F_dims) for i in range(self.num_experts)])
        self.item_f = nn.ModuleList([Expert(F_dims) for i in range(self.num_experts)])

        self.user_v = nn.Sequential(nn.Linear(self.teacher_dim, self.num_experts), nn.Softmax(dim=1))
        self.item_v = nn.Sequential(nn.Linear(self.teacher_dim, self.num_experts), nn.Softmax(dim=1))

        self.sm = nn.Softmax(dim=1)

    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]
    
    def do_something_in_each_epoch(self, epoch):
        self.T = 1.0 * ((1e-10 / 1.0) ** (epoch / self.max_epoch))
        self.T = max(self.T, 1e-10)

    def sim(self, A, B, is_inner=False):
        if not is_inner:
            denom_A = 1 / (A ** 2).sum(1, keepdim=True).sqrt()
            denom_B = 1 / (B.T ** 2).sum(0, keepdim=True).sqrt()

            sim_mat = torch.mm(A, B.T) * denom_A * denom_B
        else:
            sim_mat = torch.mm(A, B.T)
        return sim_mat

    def get_group_result(self, batch_entity, is_user=True):
        with torch.no_grad():
            if is_user:
                t = self.teacher.get_user_embedding(batch_entity)
                v = self.user_v
            else:
                t = self.teacher.get_item_embedding(batch_entity)	
                v = self.item_v

            z = v(t).max(-1)[1] 
            if not is_user:
                z = z + self.num_experts
                
            return z

    # For Adaptive Group Assignment
    def get_GA_loss(self, batch_entity, is_user=True):

        if is_user:
            s = self.student.get_user_embedding(batch_entity)													
            t = self.teacher.get_user_embedding(batch_entity)										

            f = self.user_f
            v = self.user_v
        else:
            s = self.student.get_item_embedding(batch_entity)													
            t = self.teacher.get_item_embedding(batch_entity)											
    
            f = self.item_f
            v = self.item_v

        alpha = v(t)
        g = torch.distributions.Gumbel(0, 1).sample(alpha.size()).cuda()
        alpha = alpha + 1e-10
        z = self.sm((alpha.log() + g) / self.T)

        z = torch.unsqueeze(z, 1)
        z = z.repeat(1, self.teacher_dim, 1)

        f_hat = [f[i](s).unsqueeze(-1) for i in range(self.num_experts)]
        f_hat = torch.cat(f_hat, -1)
        f_hat = f_hat * z
        f_hat = f_hat.sum(2)

        GA_loss = ((t - f_hat) ** 2).sum(-1).sum()

        return GA_loss
    
    def get_TD_loss(self, batch_user, batch_item):
        if self.choice == 'first':
            return self.get_TD_loss1(batch_user, batch_item)
        else:
            return self.get_TD_loss2(batch_user, batch_item)
                
    # Topology Distillation Loss (with Group(P,P))
    def get_TD_loss1(self, batch_user, batch_item):

        s = torch.cat([self.student.get_user_embedding(batch_user), self.student.get_item_embedding(batch_item)], 0)
        t = torch.cat([self.teacher.get_user_embedding(batch_user), self.teacher.get_item_embedding(batch_item)], 0)
        z = torch.cat([self.get_group_result(batch_user, is_user=True), self.get_group_result(batch_item, is_user=False)], 0)
        G_set = z.unique()
        Z = F.one_hot(z).float()	

        # Compute Prototype
        with torch.no_grad():
            tmp = Z.T
            tmp = tmp / (tmp.sum(1, keepdims=True) + 1e-10)
            P_s = tmp.mm(s)[G_set]
            P_t = tmp.mm(t)[G_set]

        # entity_level topology
        entity_mask = Z.mm(Z.T)        
        
        t_sim_tmp = self.sim(t, t) * entity_mask
        t_sim_dist = t_sim_tmp[t_sim_tmp > 0.]
        
        s_sim_dist = self.sim(s, s) * entity_mask    
        s_sim_dist = s_sim_dist[t_sim_tmp > 0.]
         
        # # Group_level topology
        t_proto_dist = self.sim(P_t, P_t).view(-1)
        s_proto_dist = self.sim(P_s, P_s).view(-1)

        total_loss = ((s_sim_dist - t_sim_dist) ** 2).sum() + ((s_proto_dist - t_proto_dist) ** 2).sum()

        return total_loss


    # Topology Distillation Loss (with Group(P,e))
    def get_TD_loss2(self, batch_user, batch_item):

        s = torch.cat([self.student.get_user_embedding(batch_user), self.student.get_item_embedding(batch_item)], 0)
        t = torch.cat([self.teacher.get_user_embedding(batch_user), self.teacher.get_item_embedding(batch_item)], 0)
        z = torch.cat([self.get_group_result(batch_user, is_user=True), self.get_group_result(batch_item, is_user=False)], 0)
        G_set = z.unique()
        Z = F.one_hot(z).float()

        # Compute Prototype
        with torch.no_grad():
            tmp = Z.T
            tmp = tmp / (tmp.sum(1, keepdims=True) + 1e-10)
            P_s = tmp.mm(s)[G_set]
            P_t = tmp.mm(t)[G_set]

        # entity_level topology
        entity_mask = Z.mm(Z.T)
        
        t_sim_tmp = self.sim(t, t) * entity_mask
        t_sim_dist = t_sim_tmp[t_sim_tmp > 0.]
        
        s_sim_dist = self.sim(s, s) * entity_mask    
        s_sim_dist = s_sim_dist[t_sim_tmp > 0.]
         
        # # Group_level topology 
        # t_proto_dist = (sim(P_t, t) * (1 - Z.T)[G_set]).view(-1)
        # s_proto_dist = (sim(P_s, s) * (1 - Z.T)[G_set]).view(-1)

        t_proto_dist = self.sim(P_t, t).view(-1)
        s_proto_dist = self.sim(P_s, s).view(-1)

        total_loss = ((s_sim_dist - t_sim_dist) ** 2).sum() + ((s_proto_dist - t_proto_dist) ** 2).sum()

        return total_loss
    
    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        batch_neg_item = batch_neg_item.reshape(-1)
        ## Group Assignment
        GA_loss_user = self.get_GA_loss(batch_user.unique(), is_user=True)
        GA_loss_item = self.get_GA_loss(torch.cat([batch_pos_item, batch_neg_item], 0).unique(), is_user=False)
        GA_loss = GA_loss_user + GA_loss_item

        ## Topology Distillation
        TD_loss  = self.get_TD_loss(batch_user.unique(), torch.cat([batch_pos_item, batch_neg_item], 0).unique())
        HTD_loss = TD_loss * self.alpha + GA_loss * (1 - self.alpha)
        return HTD_loss


class UnKD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)

        self.group_num = args.unkd_group_num
        self.lamda = args.unkd_lamda
        self.n_distill = args.unkd_n_distill

        self.item_group, self.group_ratio, self.user_group_ratio, self.user_group_items = self.splitGroup()

        self.get_rank_negItems()
    
    def splitGroup(self):
        print('***begin group***')
        item_count = {}
        train_pairs = self.dataset.train_pairs
        train_users = train_pairs[:, 0]
        train_items = train_pairs[:, 1]
        listLen = len(train_pairs)
        count_sum = 0
        for i in range(listLen):
                k = train_items[i]
                if k not in item_count:
                    item_count[k] = 0
                item_count[k] += 1
                count_sum += 1
        count = sorted(item_count.items(), key=lambda x: x[1], reverse=True)
        group_aver = count_sum / self.group_num
        item_group = []
        temp_group = []
        temp_count = 0
        for l in range(len(count)):
            i, j = count[l][0], count[l][1]
            temp_group.append(i)
            temp_count += j
            if temp_count > group_aver:
                if len(temp_group) == 1:
                    item_group.append(temp_group)
                    temp_group = []
                    temp_count = 0
                    continue
                temp_group.remove(i)
                item_group.append(temp_group)
                temp_group = []
                temp_group.append(i)
                temp_count = j
        if len(temp_group) > 0:
            if temp_count > group_aver / 2:
                item_group.append(temp_group)
            else:
                if len(item_group) > 0:
                    item_group[-1].extend(temp_group)
                else:
                    item_group.append(temp_group)

        print('group_len')
        for i in range(len(item_group)):
            print(len(item_group[i]))
        cate_ratio = []
        temp = 0
        print('popualrity sum')
        for i in range(len(item_group)):
            tot = 0
            tot_n = 0
            for j in range(len(item_group[i])):
                tot += item_count[item_group[i][j]]
                tot_n += 1
            print(tot)
            cate_ratio.append(tot / tot_n)
        print(cate_ratio)
        maxP = max(cate_ratio)
        minP = min(cate_ratio)
        for i in range(len(cate_ratio)):
            cate_ratio[i] = (maxP + minP) - cate_ratio[i]
            temp += cate_ratio[i]
        for i in range(len(cate_ratio)):
            cate_ratio[i] = round((cate_ratio[i] / temp), 2)
        # cate_ratio.reverse()
        for i in range(len(cate_ratio)):
            if cate_ratio[i] < 0.1:
                cate_ratio[i] = 0.1
        print(cate_ratio)

        user_group_ratio=[[] for j in range(self.num_users)]
        user_group_items = [[] for j in range(self.num_users)]
        for i in range(self.num_users):
            user_group_items[i] = [[] for j in range(self.group_num)]
            user_group_ratio[i] = [0 for j in range(self.group_num)]
        for i in range(len(train_items)):
            for k in range(len(item_group)):
                if train_items[i] in item_group[k]:
                    user_group_ratio[train_users[i]][k] += 1
                    user_group_items[train_users[i]][k].append(train_items[i])
        print('***end group***')
        return item_group, cate_ratio, user_group_ratio, user_group_items

    def get_rank_negItems(self):
        all_ratio = 0.0
        for i in range(len(self.group_ratio)):
            self.group_ratio[i] = math.pow(self.group_ratio[i], self.lamda)
            all_ratio += self.group_ratio[i]
        for i in range(len(self.group_ratio)):
            self.group_ratio[i] = self.group_ratio[i] / all_ratio
        print(self.group_ratio)
        all_n = 0
        for i in self.group_ratio:
            all_n += round(i * self.n_distill)
        print(all_n)
        if all_n < self.n_distill:
            all_n = self.n_distill
        ranking_list = np.asarray([(1 + i) / 20 for i in range(1000)])
        ranking_list = torch.FloatTensor(ranking_list)
        ranking_list = torch.exp(-ranking_list)
        self.ranking_list = ranking_list
        self.ranking_list.requires_grad = False
        self.user_negitems = [list() for u in range(self.num_users)]

        self.pos_items = torch.zeros((self.num_users, all_n))
        self.neg_items = torch.zeros((self.num_users, all_n))
        self.item_tag = torch.zeros((self.num_users, all_n))
        for i in range(len(self.item_group)):
            cate_items = set(self.item_group[i])
            ratio = self.group_ratio[i]
            dis_n = math.ceil(self.n_distill * ratio)
            for user in range(self.num_users):
                crsNeg = set(list(self.dataset.train_dict[user]))
                neglist = list(cate_items - crsNeg)
                negItems = torch.LongTensor(neglist).cuda()
                rating = self.teacher.forward_multi_items(torch.tensor([user]).cuda(), negItems.reshape(1, -1), self.teacher).reshape((-1))
                n_rat = rating.sort(dim=-1, descending=True)[1]
                negItems = negItems[n_rat]
                self.user_negitems[user].append(negItems[:1000])

    def do_something_in_each_epoch(self, epoch):
        with torch.no_grad():
            # interesting items
            pos_items = [list() for u in range(self.num_users)]
            neg_items = [list() for u in range(self.num_users)]
            item_tag = [list() for u in range(self.num_users)]
            all_n = 0
            for i in self.group_ratio:
                all_n += round(i * self.n_distill)

            for i in range(len(self.item_group)):
                ratio = self.group_ratio[i]
                dis_n = round(ratio * self.n_distill)
                if all_n < self.n_distill:
                    tm = self.n_distill - all_n
                    if i < tm:
                        dis_n += 1
                for user in range(self.num_users):
                    temp_ranklist = self.ranking_list.clone()
                    user_weight = self.user_group_ratio[user][i]
                    negItems = self.user_negitems[user][i]

                    while True:
                        k = 0
                        samples1 = torch.multinomial(temp_ranklist[:len(negItems)], dis_n, replacement=True)
                        samples2 = torch.multinomial(temp_ranklist[:len(negItems)], dis_n, replacement=True)
                        for l in range(len(samples1)):
                            if samples1[l] < samples2[l]:
                                k += 1
                            elif samples1[l] > samples2[l]:
                                k += 1
                                samples1[l], samples2[l] = samples2[l], samples1[l]
                        if k >= dis_n:
                            pos_items[user].extend(negItems[samples1])
                            neg_items[user].extend(negItems[samples2])
                            item_tag[user].extend([user_weight] * len(samples1))
                            break
            for user in range(self.num_users):
                self.pos_items[user] = torch.Tensor(pos_items[user])
                self.neg_items[user] = torch.Tensor(neg_items[user])
                self.item_tag[user] = torch.Tensor(item_tag[user])
        self.pos_items = self.pos_items.long().cuda()
        self.neg_items = self.neg_items.long().cuda()
        self.item_tag = self.item_tag.cuda()

    def get_loss(self, batch_users, batch_pos_item, batch_neg_item):
        pos_samples = self.pos_items[batch_users]
        neg_samples = self.neg_items[batch_users]
        weight_samples = self.item_tag[batch_users]
        pos_scores_S = self.student.forward_multi_items(batch_users, pos_samples)
        neg_scores_S = self.student.forward_multi_items(batch_users, neg_samples)
        mf_loss = torch.log(torch.sigmoid(pos_scores_S - neg_scores_S) + 1e-10)
        mf_loss = torch.mean(torch.neg(mf_loss), dim=-1)
        mf_loss = torch.sum(mf_loss)
        return mf_loss


class HetComp(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.num_ckpt = args.hetcomp_num_ckpt
        self.K = args.hetcomp_K
        self.p = args.hetcomp_p
        self.alpha = args.hetcomp_alpha

        self.perms, self.top_items, self.pos_items = self.construct_teacher_trajectory()
        self.user_mask = self.student.dataset.train_mat.cuda()
        self.v_result = 0
        self.last_max_idx = np.zeros(self.num_users)
        self.last_dist = None
        self.is_first = True

    def _cmp(self, f_name):
        if "BEST_EPOCH" in f_name:
            return math.inf
        pt = re.compile(r".*EPOCH_([\d\.\-\+e]+)\.pt")
        return eval(pt.search(f_name).group(1))
    
    @torch.no_grad()
    def _get_permutations(self, model):
        training = model.training
        model.eval()
        test_loader = data.DataLoader(model.user_list, batch_size=self.args.batch_size)
        topK_items = torch.zeros((model.num_users, self.K), dtype=torch.long)
        for batch_user in test_loader:
            score_mat = model.get_ratings(batch_user)
            for idx, user in enumerate(batch_user):
                pos = model.dataset.train_dict[user.item()]
                score_mat[idx, pos] = -1e10
            _, sorted_mat = torch.topk(score_mat, k=self.K, dim=1)
            topK_items[batch_user, :] = sorted_mat.detach().cpu()
        model.train(training)
        return topK_items

    def construct_teacher_trajectory(self):
        T_dir = os.path.join("checkpoints", self.args.dataset, self.args.backbone, f"scratch-{self.teacher.embedding_dim}")
        assert os.path.exists(T_dir), f"Teacher path {T_dir} doesn't exists."
        old_state = deepcopy(self.teacher.state_dict())
        f_ckpts = []
        for f in os.scandir(T_dir):
            f_ckpts.append(f.path)
        assert len(f_ckpts) >= self.num_ckpt, "Number of checkpoints must be no less than self.num_ckpt."
        f_ckpts = sorted(f_ckpts, key=lambda x: self._cmp(x))
        perms = []
        for f_ckpt in f_ckpts:
            self.teacher.load_state_dict(torch.load(f_ckpt))
            perm = self._get_permutations(self.teacher)
            perms.append(perm.cuda())
        self.teacher.load_state_dict(old_state)

        pos_dict = self.teacher.dataset.train_dict
        pos_K = min([len(p_items) for p_items in pos_dict.values()])
        pos_items = torch.zeros((self.num_users, pos_K), dtype=torch.long)
        for u, p_items in pos_dict.items():
            pos_items[u, :pos_K] = p_items[:pos_K]
        top_items = perms[0]
        return perms, top_items, pos_items.cuda()
    
    @torch.no_grad()
    def _get_NDCG_u(self, sorted_list, teacher_t_items, k):
        top_scores = np.asarray([np.exp(-t / 10) for t in range(k)])
        top_scores = (2 ** top_scores) - 1
        
        t_items = teacher_t_items[:k].cpu()

        denom = np.log2(np.arange(2, k + 2))
        dcg = np.sum((np.in1d(sorted_list[:k], list(t_items)) * top_scores) / denom)
        idcg = np.sum((top_scores / denom)[:k])
        return round(dcg / idcg, 4)

    def _DKC(self, sorted_mat, last_max_idx, last_dist, is_first, epoch):
        next_idx = last_max_idx[:] 
        if is_first:
            last_dist = np.ones_like(next_idx)
            for user in self.student.user_list:
                next_v = min(self.num_ckpt - 1, int(next_idx[user]) + 1)

                next_perm = self.perms[next_v][user]
                next_dist = 1. - self._get_NDCG_u(sorted_mat[user], next_perm, self.K)
                
                last_dist[user] = next_dist

            return next_idx, last_dist

        th = self.alpha * (0.995 ** (epoch // self.p))

        for user in self.student.user_list:
            if int(last_max_idx[user]) == self.num_ckpt - 1:
                continue

            next_v = min(self.num_ckpt - 1, int(next_idx[user]) + 1)
            next_next_v = min(self.num_ckpt - 1, int(next_idx[user]) + 2)
            
            next_perm = self.perms[next_v][user]
            next_next_perm = self.perms[next_next_v][user]
            
            next_dist = 1. - self._get_NDCG_u(sorted_mat[user], next_perm, self.K)
            
            if (last_dist[user] / next_dist > th) or (last_dist[user] / next_dist < 1):
                next_idx[user] += 1
                next_next_dist = 1. - self._get_NDCG_u(sorted_mat[user], next_next_perm, self.K)
                last_dist[user] = next_next_dist
        return next_idx, last_dist
    
    def do_something_in_each_epoch(self, epoch):
        ### DKC
        if epoch % self.p == 0 and epoch >= self.p and self.v_result < self.num_ckpt - 1:
            sorted_mat = self._get_permutations(self.student)
            if self.is_first == True:
                self.last_max_idx, self.last_dist = self._DKC(sorted_mat, self.last_max_idx, self.last_dist, True, epoch)
                self.is_first = False
            else:
                self.last_max_idx, self.last_dist = self._DKC(sorted_mat, self.last_max_idx, self.last_dist, False, epoch)
            for user in self.student.user_list:
                self.top_items[user] = self.perms[int(self.last_max_idx[user])][user]
            self.v_result = round(self.last_max_idx.mean(), 2)
        
    def overall_loss(self, batch_full_mat, pos_items, top_items, batch_user_mask):
        tops = torch.gather(batch_full_mat, 1, torch.cat([pos_items, top_items], -1))
        tops_els = (batch_full_mat.exp() * (1 - batch_user_mask)).sum(1, keepdims=True)
        els = tops_els - torch.gather(batch_full_mat, 1, top_items).exp().sum(1, keepdims=True)
        
        above = tops.view(-1, 1)
        below = torch.cat((pos_items.size(1) + top_items.size(1)) * [els], 1).view(-1, 1) + above.exp()
        below = torch.clamp(below, 1e-5).log()

        return -(above - below).sum()

    def rank_loss(self, batch_full_mat, pos_items, top_items, batch_user_mask):
        S_pos = torch.gather(batch_full_mat, 1, pos_items)
        S_top = torch.gather(batch_full_mat, 1, top_items[:, :top_items.size(1) // 2])

        below2 = (batch_full_mat.exp() * (1 - batch_user_mask)).sum(1, keepdims=True) - S_top.exp().sum(1, keepdims=True)
        
        above_pos = S_pos.sum(1, keepdims=True)
        above_top = S_top.sum(1, keepdims=True)
        
        below_pos = S_pos.flip(-1).exp().cumsum(1)
        below_top = S_top.flip(-1).exp().cumsum(1)
        
        below_pos = (torch.clamp(below_pos + below2, 1e-5)).log().sum(1, keepdims=True)        
        below_top = (torch.clamp(below_top + below2, 1e-5)).log().sum(1, keepdims=True)  

        pos_KD_loss = -(above_pos - below_pos).sum()

        S_top_sub = torch.gather(batch_full_mat, 1, top_items[:, :top_items.size(1) // 10])
        below2_sub = (batch_full_mat.exp() * (1-batch_user_mask)).sum(1, keepdims=True) - S_top_sub.exp().sum(1, keepdims=True)
        
        above_top_sub = S_top_sub.sum(1, keepdims=True)
        below_top_sub = S_top_sub.flip(-1).exp().cumsum(1)
        below_top_sub = (torch.clamp(below_top_sub + below2_sub, 1e-5)).log().sum(1, keepdims=True)  

        top_KD_loss = - (above_top - below_top).sum() - (above_top_sub - below_top_sub).sum()

        return  pos_KD_loss + top_KD_loss / 2

    def get_loss(self, batch_users):
        batch_full_mat = torch.clamp(self.student.get_ratings(batch_users), min=-40, max=40)
        batch_user_mask = torch.index_select(self.user_mask, 0, batch_users).to_dense()
        t_items = torch.index_select(self.top_items, 0, batch_users)
        p_items = torch.index_select(self.pos_items, 0, batch_users)
        if self.v_result < self.num_ckpt - 1:
            KD_loss = self.overall_loss(batch_full_mat, p_items, t_items, batch_user_mask) * 2.
        else:
            KD_loss = self.rank_loss(batch_full_mat, p_items, t_items, batch_user_mask)
        return KD_loss

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        kd_loss = self.get_loss(batch_user)
        loss = self.lmbda * kd_loss
        return loss
