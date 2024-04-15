import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseRec(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        
        self.dataset = dataset
        self.args = args

        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items

        self.user_list = torch.LongTensor([i for i in range(self.num_users)]).cuda()
        self.item_list = torch.LongTensor([i for i in range(self.num_items)]).cuda()
    
    def forward(self):
        raise NotImplementedError
    
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
        raise NotImplementedError
    
    def get_all_ratings(self):
        raise NotImplementedError

    def get_ratings(self, batch_user):
        raise NotImplementedError
    
    def get_all_embedding(self):
        raise NotImplementedError
    
    def get_loss(self, output):
        """Compute the loss function with the model output

        Parameters
        ----------
        output : 
            model output (results of forward function)

        Returns
        -------
        loss : float
        """
        pos_score, neg_score = output[0], output[1]
        loss = -F.logsigmoid(pos_score - neg_score).sum()
        return loss


class BaseGCN(BaseRec):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
    
    def computer(self):
        """
        propagate methods
        """
        raise NotImplementedError

    def get_all_pre_embedding(self):
        """get total embedding of users and items before convolution

        Returns
        -------
        users : 2-D FloatTensor (num. users x dim)
        items : 2-D FloatTensor (num. items x dim)
        """
        raise NotImplementedError
    
    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        """
        Parameters
        ----------
        batch_user : 1-D LongTensor (batch_size)
        batch_pos_item : 1-D LongTensor (batch_size)
        batch_neg_item : 1-D LongTensor (batch_size)

        Returns
        -------
        output : 
            Model output to calculate its loss function
        """
        all_users, all_items = self.computer()

        u = all_users[batch_user]
        i = all_items[batch_pos_item]
        j = all_items[batch_neg_item]
        
        pos_score = (u * i).sum(dim=1, keepdim=True)
        neg_score = (u * j).sum(dim=1, keepdim=True)

        return pos_score, neg_score
    
    def get_user_embedding(self, batch_user):
        all_users, all_items = self.computer()
        users = all_users[batch_user]
        return users
    
    def get_item_embedding(self, batch_item):
        all_users, all_items = self.computer()
        items = all_items[batch_item]
        return items

    def get_all_post_embedding(self):
        """get total embedding of users and items after convolution

        Returns
        -------
        users : 2-D FloatTensor (num. users x dim)
        items : 2-D FloatTensor (num. items x dim)
        """
        all_users, all_items = self.computer()
        return all_users, all_items

    def get_all_embedding(self):
        return self.get_all_post_embedding()
    
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

        batch_user = batch_user.unsqueeze(-1)
        batch_user = torch.cat(batch_items.size(1) * [batch_user], 1)
        
        u = all_users[batch_user]		# batch_size x k x dim
        i = all_items[batch_items]		# batch_size x k x dim
        
        score = (u * i).sum(dim=-1, keepdim=False)
        
        return score
    
    def get_all_ratings(self):
        users, items = self.get_all_post_embedding()
        score_mat = torch.matmul(users, items.T)
        return score_mat

    def get_ratings(self, batch_user):
        users, items = self.get_all_post_embedding()
        users = users[batch_user]
        score_mat = torch.matmul(users, items.T)
        return score_mat
