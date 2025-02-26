# Build upon https://github.com/enoche/MMRec
import os
from torch_scatter import scatter_add

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseMM
from .base_layer import BipartitleGraph
from .utils import EmbLoss, load_pkls, dump_pkls


class VBPR(BaseMM):
    def __init__(self, dataset, mm_dict, args):
        super().__init__(dataset, mm_dict, args)
        self.model_name = "vbpr"
        self.embedding_dim = args.embedding_dim
        self.reg_weight = args.reg_weight

        self.user_id_emb = nn.Embedding(self.num_users, self.embedding_dim * 2)
        self.item_id_emb = nn.Embedding(self.num_items, self.embedding_dim)
        self.item_raw_features = torch.concat(list(mm_dict.values()), dim=-1).cuda()
        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.embedding_dim)
        self.reg_loss = EmbLoss()

        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def get_user_embedding(self, batch_user):
        users = self.user_id_emb(batch_user)
        return users
    
    def get_item_embedding(self, batch_item):
        id_embed = self.item_id_emb(batch_item)
        modal_embed = self.item_linear(self.item_raw_features[batch_item])
        item_embed = torch.cat([id_embed, modal_embed], dim=-1)
        return item_embed

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        u = self.get_user_embedding(batch_user)
        i = self.get_item_embedding(batch_pos_item)
        j = self.get_item_embedding(batch_neg_item)
        pos_score = (u * i).sum(dim=1, keepdim=True)    # batch_size, 1
        neg_score = torch.bmm(j, u.unsqueeze(-1)).squeeze(-1)       # batch_size, num_ns
        reg_loss = self.reg_loss(u, i, j)
        return pos_score, neg_score, reg_loss

    def get_loss(self, output):
        pos_score, neg_score, reg_loss = output
        pos_score = pos_score.expand_as(neg_score)  # batch_size, num_ns
        mf_loss = -F.logsigmoid(pos_score - neg_score).mean(1).sum()
        loss = mf_loss + self.reg_weight * reg_loss
        return loss
    
    def get_all_embedding(self):
        users = self.user_id_emb.weight
        items = self.get_item_embedding(self.item_list)
        return users, items
    
    def get_item_modality_embedding(self, batch_item):
        modal_embed = self.item_linear(self.item_raw_features[batch_item])
        mm_feat_dict = {"mixed": modal_embed}
        return mm_feat_dict
    
    def forward_multi_items(self, batch_user, batch_items):
        u = self.get_user_embedding(batch_user)		# batch_size x dim
        i = self.get_item_embedding(batch_items)		# batch_size x k x dim
        score = torch.bmm(i, u.unsqueeze(-1)).squeeze(-1)   # batch_size, k
        return score
    
    def get_all_ratings(self):
        users, items = self.get_all_embedding()
        score_mat = torch.matmul(users, items.T)
        return score_mat

    def get_ratings(self, batch_user):
        users = self.get_user_embedding(batch_user.cuda())
        items = self.get_item_embedding(self.item_list)
        score_mat = torch.matmul(users, items.T)
        return score_mat


class BM3(BaseMM):
    def __init__(self, dataset, mm_dict, args):
        super().__init__(dataset, mm_dict, args)
        self.model_name = "bm3"
        self.embedding_dim = args.embedding_dim
        self.feat_embed_dim = args.embedding_dim
        self.n_layers = args.n_layers
        self.reg_weight = args.reg_weight
        self.cl_weight = args.cl_weight
        self.dropout = args.dropout

        self.modality_names = list(mm_dict.keys())
        self.Graph = BipartitleGraph(args, dataset).graph
        self.user_id_emb = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_id_emb = nn.Embedding(self.num_items, self.embedding_dim)
        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()
        self.mm_emb_dict = nn.ModuleDict({
            name: nn.Embedding.from_pretrained(data, freeze=False) for (name, data) in mm_dict.items()
        })
        self.mm_trs_dict = nn.ModuleDict({
            name: nn.Linear(data.shape[1], self.feat_embed_dim) for (name, data) in mm_dict.items()
        })
        self.reset_para()

    def reset_para(self):
        nn.init.xavier_uniform_(self.user_id_emb.weight)
        nn.init.xavier_uniform_(self.item_id_emb.weight)
        nn.init.xavier_normal_(self.predictor.weight)
        for m in self.mm_trs_dict:
            nn.init.xavier_normal_(self.mm_trs_dict[m].weight)
    
    def computer(self):
        h = self.item_id_emb.weight
        ego_embeddings = torch.cat((self.user_id_emb.weight, self.item_id_emb.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.Graph, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h
    
    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        u_online_ori, i_online_ori = self.computer()
        u_online_ori = u_online_ori[batch_user]
        i_online_ori = i_online_ori[batch_pos_item]
        mm_onlines, mm_targets = {}, {}
        for m in self.mm_emb_dict:
            trs = self.mm_trs_dict[m]
            feat = self.mm_emb_dict[m](batch_pos_item)
            mm_onlines[m] = trs(feat)
        with torch.no_grad():
            u_target, i_target = u_online_ori.clone().detach(), i_online_ori.clone().detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)
            for m in mm_onlines:
                mm_target = mm_onlines[m].clone()
                mm_targets[m] = F.dropout(mm_target, self.dropout)
        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)
        loss_mi, loss_mm = 0., 0.
        for m in mm_onlines:
            mm_online = self.predictor(mm_onlines[m])
            mm_target = mm_targets[m]
            loss_mi += 1. - F.cosine_similarity(mm_online, i_target.detach(), dim=-1).mean()
            loss_mm += 1. - F.cosine_similarity(mm_online, mm_target.detach(), dim=-1).mean()
        loss_ui = 1. - F.cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1. - F.cosine_similarity(i_online, u_target.detach(), dim=-1).mean()
        loss_reg = self.reg_loss(u_online_ori, i_online_ori)
        return loss_ui, loss_iu, loss_mi, loss_mm, loss_reg
    
    def get_loss(self, output):
        loss_ui, loss_iu, loss_mi, loss_mm, loss_reg = output
        loss = (loss_ui + loss_iu) + self.reg_weight * loss_reg + \
               self.cl_weight * (loss_mi + loss_mm)
        return loss
    
    def get_user_embedding(self, batch_user):
        all_users, all_items = self.computer()
        users = self.predictor(all_users[batch_user])
        return users
    
    def get_item_embedding(self, batch_item):
        all_users, all_items = self.computer()
        items = self.predictor(all_items[batch_item])
        return items

    def get_all_embedding(self):
        all_users, all_items = self.computer()
        users = self.predictor(all_users)
        items = self.predictor(all_items)
        return users, items
    
    def get_item_modality_embedding(self, batch_item):
        mm_feat_dict = {}
        for modality in self.mm_emb_dict:
            trs = self.mm_trs_dict[modality]
            feat = self.mm_emb_dict[modality](batch_item)
            mm_feat_dict[modality] = trs(feat)
        return mm_feat_dict
    
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
        all_users, all_items = self.computer()
        
        u = self.predictor(all_users[batch_user])		# batch_size x dim
        i = self.predictor(all_items[batch_items])		# batch_size x k x dim
        
        score = torch.bmm(i, u.unsqueeze(-1)).squeeze(-1)   # batch_size, k
        
        return score
    
    def get_all_ratings(self):
        users, items = self.get_all_embedding()
        score_mat = torch.matmul(users, items.T)
        return score_mat

    def get_ratings(self, batch_user):
        users, items = self.get_all_embedding()
        users = users[batch_user]
        score_mat = torch.matmul(users, items.T)
        return score_mat


class MGCN(BaseMM):
    def __init__(self, dataset, mm_dict, args):
        super().__init__(dataset, mm_dict, args)
        self.model_name = "mgcn"
        self.embedding_dim = args.embedding_dim
        self.n_ui_layers = args.n_ui_layers
        self.knn_k = args.knn_k
        self.n_mm_layers = args.n_mm_layers
        self.reg_weight = args.reg_weight
        self.cl_weight = args.cl_weight
        self.tau = 0.2

        self.user_id_embed = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_id_embed = nn.Embedding(self.num_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_id_embed.weight)
        nn.init.xavier_uniform_(self.item_id_embed.weight)

        self.norm_adj = BipartitleGraph(args, dataset).graph
        indices = torch.arange(self.num_users).long().cuda()
        self.R = self.norm_adj.index_select(dim=0, index=indices)
        indices = torch.arange(self.num_items).long().cuda() + self.num_users
        self.R = self.R.index_select(dim=1, index=indices)
        self.mm_adj_dict = self.load_mm_adj_dict(mm_dict)
        self.softmax = nn.Softmax(dim=-1)
        
        self.mm_emb_dict = nn.ModuleDict({
            name: nn.Embedding.from_pretrained(data, freeze=False) for (name, data) in mm_dict.items()
        })
        self.mm_trs_dict = nn.ModuleDict({
            name: nn.Linear(data.shape[1], self.embedding_dim) for (name, data) in mm_dict.items()
        })
        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )
        self.mm_gate_dict = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Sigmoid()
            ) for name in mm_dict
        })
        self.mm_gate_prefer_dict = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Sigmoid()
            ) for name in mm_dict
        })

    def get_sparse_laplacian(self, edge_index, edge_weight, num_nodes, normalization='none'):
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        if normalization == 'sym':
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        elif normalization == 'rw':
            deg_inv = 1.0 / deg
            deg_inv.masked_fill_(deg_inv == float('inf'), 0)
            edge_weight = deg_inv[row] * edge_weight
        return edge_index, edge_weight

    def load_mm_adj_dict(self, mm_dict):
        mm_adj_dict = {}
        for m in mm_dict:
            f_adj = os.path.join("modeling", "backbone", "crafts", self.args.dataset, f"{self.model_name}", f"KNNGraph_{self.knn_k}_{m}.pkl")
            sucflg, adj = load_pkls(f_adj)
            if sucflg:
                mm_adj_dict[m] = adj.cuda()
                continue
            context = mm_dict[m]
            context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
            adj = torch.mm(context_norm, context_norm.transpose(1, 0))
            knn_val, knn_ind = torch.topk(adj, self.knn_k, dim=-1)
            tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
            row = [i[0] for i in tuple_list]
            col = [i[1] for i in tuple_list]
            i = torch.LongTensor([row, col]).cuda()
            v = knn_val.flatten().cuda()
            edge_index, edge_weight = self.get_sparse_laplacian(i, v, normalization="sym", num_nodes=adj.shape[0])
            adj = torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
            dump_pkls((adj.cpu(), f_adj))
            mm_adj_dict[m] = adj.cuda()
        return mm_adj_dict

    def computer(self):
        # User-Item View
        item_embeds = self.item_id_embed.weight
        user_embeds = self.user_id_embed.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        content_embeds = all_embeddings.mean(dim=1, keepdim=False)

        # Item-Item View
        mm_embeds_dict = {}
        for m in self.mm_emb_dict:
            mm_feat = self.mm_trs_dict[m](self.mm_emb_dict[m].weight)
            item_embeds = torch.multiply(self.item_id_embed.weight, self.mm_gate_dict[m](mm_feat))
            for i in range(self.n_mm_layers):
                item_embeds = torch.sparse.mm(self.mm_adj_dict[m], item_embeds)
            user_embeds = torch.sparse.mm(self.R, item_embeds)
            mm_embeds_dict[m] = torch.cat([user_embeds, item_embeds], dim=0)

        # Behavior-Aware Fuser
        att_common = torch.cat([self.query_common(embeds) for embeds in mm_embeds_dict.values()], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeds = 0.
        for idx, m in enumerate(self.mm_emb_dict):
            common_embeds += weight_common[:, idx].unsqueeze(dim=1) * mm_embeds_dict[m]
        mm_sep_dict = {}
        for m in mm_embeds_dict:
            sep_embeds = mm_embeds_dict[m] - common_embeds
            prefer = self.mm_gate_prefer_dict[m](content_embeds)
            mm_sep_dict[m] = torch.multiply(prefer, sep_embeds)
        side_embeds = common_embeds
        for m in mm_sep_dict:
            side_embeds += mm_sep_dict[m]
        side_embeds /= (len(mm_sep_dict) + 1)

        all_embeds = content_embeds + side_embeds
        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.num_users, self.num_items], dim=0)
        return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.sum(cl_loss)

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.computer()

        u = ua_embeddings[batch_user]
        i = ia_embeddings[batch_pos_item]
        j = ia_embeddings[batch_neg_item]
        pos_score = (u * i).sum(dim=1, keepdim=True)    # batch_size, 1
        neg_score = torch.bmm(j, u.unsqueeze(-1)).squeeze(-1)
        pos_score = pos_score.expand_as(neg_score)  # batch_size, num_ns
        bpr_loss = -F.logsigmoid(pos_score - neg_score).mean(1).sum()

        reg_loss = 1. / 2 * (u ** 2).sum() + 1. / 2 * (i ** 2).sum() + 1. / 2 * (j ** 2).sum()

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.num_users, self.num_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.num_users, self.num_items], dim=0)
        cl_loss_item = self.InfoNCE(side_embeds_items[batch_pos_item], content_embeds_items[batch_pos_item], self.tau)
        cl_loss_user = self.InfoNCE(side_embeds_users[batch_user], content_embeds_user[batch_user], self.tau)
        cl_loss = cl_loss_item + cl_loss_user

        return bpr_loss, reg_loss, cl_loss
    
    def get_loss(self, output):
        bpr_loss, reg_loss, cl_loss = output
        loss = bpr_loss + self.reg_weight * reg_loss + self.cl_weight * cl_loss
        return loss
    
    def get_user_embedding(self, batch_user):
        all_users, all_items, side_embeds, content_embeds = self.computer()
        users = all_users[batch_user]
        return users
    
    def get_item_embedding(self, batch_item):
        all_users, all_items, side_embeds, content_embeds = self.computer()
        items = all_items[batch_item]
        return items

    def get_all_embedding(self):
        all_users, all_items, side_embeds, content_embeds = self.computer()
        return all_users, all_items
    
    def get_item_modality_embedding(self, batch_item):
        all_users, all_items, side_embeds, content_embeds = self.computer()
        mm_feat_dict = {"fused": side_embeds}
        return mm_feat_dict
    
    def forward_multi_items(self, batch_user, batch_items):
        all_users, all_items = self.get_all_embedding()
        u = all_users[batch_user]		# batch_size x dim
        i = all_items[batch_items]		# batch_size x k x dim
        score = torch.bmm(i, u.unsqueeze(-1)).squeeze(-1)   # batch_size, k
        return score
    
    def get_all_ratings(self):
        users, items = self.get_all_embedding()
        score_mat = torch.matmul(users, items.T)
        return score_mat

    def get_ratings(self, batch_user):
        users, items = self.get_all_embedding()
        users = users[batch_user]
        score_mat = torch.matmul(users, items.T)
        return score_mat
