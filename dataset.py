import os
import yaml
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F


DATA_DIR = 'data/'


def load_pii_dict(file_path, start_idx):
    ui_pairs = []
    uis_dict = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            u, i = line.strip().split()
            u = int(u) - start_idx
            i = int(i) - start_idx
            ui_pairs.append([u, i])
            if u not in uis_dict:
                uis_dict[u] = []
            uis_dict[u].append(i)

    ui_pairs = torch.LongTensor(ui_pairs)
    for u in uis_dict:
        uis_dict[u] = torch.LongTensor(uis_dict[u])

    return ui_pairs, uis_dict


def load_data(dataset_name):
    """
    load raw data (u-i pairs) from train.txt, valid.txt, test.txt, config.yaml
    all indexes start from 0
    """
    data_dir = os.path.join(DATA_DIR, dataset_name)
    config = yaml.load(open(os.path.join(data_dir, 'config.yaml'), 'r'), Loader=yaml.FullLoader)

    start_idx = config['start_idx']
    num_users = config['num_users']
    num_items = config['num_items']

    train_pairs, train_dict = load_pii_dict(os.path.join(data_dir, 'train.txt'), start_idx)
    valid_pairs, valid_dict = load_pii_dict(os.path.join(data_dir, 'valid.txt'), start_idx)
    test_pairs, test_dict = load_pii_dict(os.path.join(data_dir, 'test.txt'), start_idx)

    # Define train_matrix with sparse matrix
    index = train_pairs.t()
    data = torch.ones(index.size(-1)).int()
    train_matrix = torch.sparse_coo_tensor(index, data,
                                        torch.Size([num_users, num_items]), dtype=torch.int)
    train_user_pop = torch.sparse.sum(train_matrix, dim=1).to_dense()
    train_item_pop = torch.sparse.sum(train_matrix, dim=0).to_dense()
    
    return num_users, num_items, train_pairs, valid_pairs, test_pairs, train_dict, valid_dict, test_dict, train_matrix, train_user_pop, train_item_pop


#################################################################################################################
# For training
#################################################################################################################

class implicit_CF_dataset(data.Dataset):

    def __init__(self, dataset, num_users, num_items, train_pairs, train_mat, train_dict, user_pop, item_pop, num_ns):
        """
        Parameters
        ----------
        dataset: str
            name of dataset
        num_users : int
            num. users
        num_items : int
            num. items
        train_pairs : torch.Tensor (num_train_pairs, 2)
            total train train_pairs, each instance has a form of (user, item)
        train_mat : torch.sparse_coo_tensor (num_users, num_items)
            user-item training rating matrix being 0 or 1
        train_dict: dict
            user as keys, interacted item lists as values
        user_pop: torch.Tensor (num_users)
            popularity of each user
        item_pop: torch.Tensor (num_items)
            popularity of each item
        num_ns : int
            num. negative samples
        """
        super(implicit_CF_dataset, self).__init__()
        
        self.dataset = dataset
        self.num_users = num_users
        self.num_items = num_items
        self.train_pairs = train_pairs
        self.train_mat = train_mat
        self.train_dict = train_dict
        self.user_pop = user_pop
        self.item_pop = item_pop
        self.num_ns = num_ns

        self.users, self.pos_items, self.neg_items = None, None, None

    def negative_sampling(self):
        """conduct the negative sampling
        """
        users = []
        pos_items = []
        neg_items = []
        for user, pos in self.train_dict.items():
            pos = pos.numpy()
            users.append(np.array([user]).repeat(len(pos)))
            pos_items.append(pos)
            neg_candidates = np.arange(self.num_items)
            probs = np.ones(self.num_items)
            probs[pos] = 0
            probs /= np.sum(probs)
            neg = np.random.choice(neg_candidates, size=len(pos) * self.num_ns, p=probs, replace=True).reshape(len(pos), self.num_ns)
            neg_items.append(neg)
        users = np.concatenate(users, axis=0)
        pos_items = np.concatenate(pos_items, axis=0)
        neg_items = np.concatenate(neg_items, axis=0)
        self.users = torch.from_numpy(users)
        self.pos_items = torch.from_numpy(pos_items)
        self.neg_items = torch.from_numpy(neg_items)
    
    def __len__(self):
        return len(self.train_pairs)
    
    def __getitem__(self, idx):
        return self.users[idx], self.pos_items[idx], self.neg_items[idx]
        

#################################################################################################################
# For test
#################################################################################################################

class implicit_CF_dataset_test(data.Dataset):
    def __init__(self, num_users, num_items, valid_dict, test_dict):
        """
        Parameters
        ----------
        num_users : int
            num. users
        num_items : int
            num. items
        valid_dict: dict
            user as keys, valid item as values
        test_dict: dict
            user as keys, test item as values
        """
        super(implicit_CF_dataset_test, self).__init__()

        self.user_num = num_users
        self.item_num = num_items
        self.user_list = torch.LongTensor([i for i in range(num_users)])

        self.valid_dict = valid_dict
        self.test_dict = test_dict
