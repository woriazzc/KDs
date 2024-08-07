import os
import yaml
import math
import lmdb
import pickle
import shutil
import struct
import numpy as np
from pathlib import Path
from functools import lru_cache
from collections import defaultdict

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
            u, i = line.strip().split()[:2]
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


def load_cf_data(dataset_name):
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

    def __init__(self, dataset, num_users, num_items, train_pairs, train_mat, train_dict, user_pop, item_pop, num_ns, neg_sampling_on_all=False):
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
        neg_sampling_on_all: Bool
            if True, don't ignore positive items when negative sampling (defalut: False)
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
        self.neg_sampling_on_all = neg_sampling_on_all

        self.users, self.pos_items, self.neg_items = None, None, None

    def negative_sampling(self):
        """conduct the negative sampling
        """
        users = []
        pos_items = []
        neg_items = []
        if self.neg_sampling_on_all:
            users, pos_items = self.train_pairs[:, 0].numpy(), self.train_pairs[:, 1].numpy()
            neg_items = np.random.choice(self.num_items, size=(self.train_pairs.size(0), self.num_ns), replace=True)
        else:
            for user, pos in self.train_dict.items():
                pos = pos.numpy()
                users.append(np.array([user]).repeat(len(pos)))
                pos_items.append(pos)
                probs = np.ones(self.num_items)
                probs[pos] = 0
                probs /= np.sum(probs)
                neg = np.random.choice(self.num_items, size=len(pos) * self.num_ns, p=probs, replace=True).reshape(len(pos), self.num_ns)
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
    def __init__(self, num_users, num_items, inter_dict):
        """
        Parameters
        ----------
        num_users : int
            num. users
        num_items : int
            num. items
        inter_dict: dict
            user as keys, valid item as values
        test_dict: dict
            user as keys, test item as values
        """
        super(implicit_CF_dataset_test, self).__init__()

        self.user_num = num_users
        self.item_num = num_items
        self.user_list = torch.LongTensor([i for i in range(num_users)])

        self.inter_dict = inter_dict



#################################################################################################################
# CTR datasets
#################################################################################################################

class CTRDataset(data.Dataset):
    """
    Data prepration:
        * Remove the infrequent features (appearing in less than threshold instances) and treat them as a single feature
        * Discretize numerical values by log2 transformation which is proposed by the winner of Criteo Competition
        * Must put train.csv, valid.csv, test.csv in data_dir

    :param dataset_name
    """
    def __init__(self, dataset_name, mode):
        dataset_name = dataset_name.lower()
        self.data_dir = os.path.join(DATA_DIR, dataset_name)
        config = yaml.load(open(os.path.join(self.data_dir, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
        self.header = config['header']
        self.sep = config['sep']
        self.num_feats = config["num_feats"]
        self.num_int_feats = config["num_int_feats"]
        self.min_threshold = config["min_threshold"]
        cache_dir = os.path.join(self.data_dir, f".{mode}")
        data_path = os.path.join(self.data_dir, f"{mode}.csv")
        if not Path(cache_dir).exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            self.__build_cache(data_path, cache_dir)

        self.env = lmdb.open(cache_dir, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
            self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(
                txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.int64)
        #     x, y
        return np_array[1:], np_array[0]

    def __len__(self):
        return self.length

    def __build_cache(self, path, cache_path):
        temp_path = os.path.join(self.data_dir, "train.csv")
        # count feature map
        feat_mapper, defaults = self.__get_feat_mapper(temp_path)
        # load feature map
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            field_dims = np.zeros(self.num_feats, dtype=np.uint32)
            for i, fm in feat_mapper.items():
                field_dims[i - 1] = len(fm) + 1

            # write field_dims
            with env.begin(write=True) as txn:
                txn.put(b'field_dims', field_dims.tobytes())

            for buffer in self.__yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)

    def __get_feat_mapper(self, path):
        feat_cnts = defaultdict(lambda: defaultdict(int))
        with open(path, 'r') as f:
            first_flg = self.header
            for line in f:
                if first_flg:
                    first_flg = False
                    continue

                values = line.rstrip('\n').split(self.sep)
                if len(values) != self.num_feats + 1:
                    continue

                for i in range(1, self.num_int_feats + 1):
                    feat_cnts[i][convert_numeric_feature(values[i])] += 1

                for i in range(self.num_int_feats + 1, self.num_feats + 1):
                    feat_cnts[i][values[i]] += 1

        feat_mapper = {i: {feat for feat, c in cnt.items() if c >= self.min_threshold} for i, cnt in feat_cnts.items()}
        feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}

        return feat_mapper, defaults

    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path, 'r') as f:
            first_flg = self.header
            for line in f:
                if first_flg:
                    first_flg = False
                    continue
                values = line.rstrip('\n').split(self.sep)
                if len(values) != self.num_feats + 1:
                    continue
                np_array = np.zeros(self.num_feats + 1, dtype=np.uint32)
                np_array[0] = int(values[0])
               
                for i in range(1, self.num_int_feats + 1):
                    np_array[i] = feat_mapper[i].get(convert_numeric_feature(values[i]), defaults[i])

                for i in range(self.num_int_feats + 1, self.num_feats + 1):
                    np_array[i] = feat_mapper[i].get(values[i], defaults[i])
                buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
            yield buffer


@lru_cache(maxsize=None)
def convert_numeric_feature(val: str):
    if val == '':
        return 'NULL'
    v = int(val)
    if v > 2:
        return str(int(math.log(v) ** 2))
    else:
        return str(v)
