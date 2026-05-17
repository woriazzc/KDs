import os
import re
import yaml
import math
import pickle
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from .parse import DATA_DIR


def load_multimodal(dataset_name):
    data_dir = os.path.join(DATA_DIR, dataset_name)
    mm_dict = {}
    for f_name in os.listdir(data_dir):
        mm_name = re.match(r"(.+)_modality.pkl", f_name)
        if mm_name is None: continue
        mm_name = mm_name.group(1)
        mm_data = pickle.load(open(os.path.join(data_dir, f_name), "rb"))
        mm_dict[mm_name] = mm_data  # torch.Tensor, device=cpu, dtype=float32
    return mm_dict


def load_pii_dict(file_path, start_idx):
    df = pd.read_csv(file_path, sep='\t', header=0)
    users = df["USER"].to_numpy() - start_idx
    items = df["ITEM"].to_numpy() - start_idx
    ui_pairs = np.stack([users, items], axis=-1)
    
    uis_dict = {}
    for pii in ui_pairs:
        u, i = pii[0], pii[1]
        if u not in uis_dict:
            uis_dict[u] = []
        uis_dict[u].append(i)
    
    ui_pairs = torch.from_numpy(ui_pairs).long()
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

    def __init__(self, dataset, num_users, num_items, train_pairs, train_mat, train_dict, user_pop, item_pop, num_ns, neg_sampling_on_all=False, no_neg_sampling=False):
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
            if True, don't ignore positive items when negative sampling (default: False)
        no_neg_sampling: Bool
            if True, return zero vectors (default: False)
        """
        super().__init__()
        
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
        self.no_neg_sampling = no_neg_sampling

        self.users, self.pos_items, self.neg_items = None, None, None

    def negative_sampling(self):
        """conduct the negative sampling
        """
        if self.no_neg_sampling:
            users, pos_items = self.train_pairs[:, 0].numpy(), self.train_pairs[:, 1].numpy()
            neg_items = np.zeros((self.train_pairs.size(0), self.num_ns))
            self.users = torch.from_numpy(users)
            self.pos_items = torch.from_numpy(pos_items)
            self.neg_items = torch.from_numpy(neg_items)
            return
        if self.neg_sampling_on_all:
            users, pos_items = self.train_pairs[:, 0].numpy(), self.train_pairs[:, 1].numpy()
            neg_items = np.random.choice(self.num_items, size=(self.train_pairs.size(0), self.num_ns), replace=True)
        else:
            users = []
            pos_items = []
            neg_items = []
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


def split_implicit_CF_dataset(dataset_fa:implicit_CF_dataset, ratio:float=0.9):
    train_dict_son = {}
    train_dict_fa = dataset_fa.train_dict
    train_list_son = []
    for u in train_dict_fa:
        train_dict_son[u] = []
        items_fa = train_dict_fa[u]
        count_son = math.ceil(len(items_fa) * ratio)
        for idx in range(count_son):
            train_dict_son[u].append(train_dict_fa[u][idx])
            train_list_son.append([u, train_dict_fa[u][idx]])
        train_dict_son[u] = torch.LongTensor(train_dict_son[u])
    train_pair_son = torch.LongTensor(train_list_son)
    # Define train_matrix with sparse matrix
    index = train_pair_son.t()
    data = torch.ones(index.size(-1)).int()
    train_matrix_son = torch.sparse_coo_tensor(index, data,
                                        torch.Size([dataset_fa.num_users, dataset_fa.num_items]), dtype=torch.int)
    train_user_pop_son = torch.sparse.sum(train_matrix_son, dim=1).to_dense()
    train_item_pop_son = torch.sparse.sum(train_matrix_son, dim=0).to_dense()
    dataset_son = implicit_CF_dataset(
        dataset_fa.dataset,
        dataset_fa.num_users,
        dataset_fa.num_items,
        train_pair_son,
        train_matrix_son,
        train_dict_son,
        train_user_pop_son,
        train_item_pop_son,
        dataset_fa.num_ns,
        dataset_fa.neg_sampling_on_all,
        dataset_fa.no_neg_sampling
    )
    return dataset_son


class implicit_SR_dataset(data.Dataset):
    def __init__(self, CF_dataset: implicit_CF_dataset, max_sequence_len):
        super().__init__()
        self.dataset = CF_dataset.dataset
        self.train_dict = deepcopy(CF_dataset.train_dict)
        self.num_users = CF_dataset.num_users
        self.num_items = CF_dataset.num_items
        self.num_seqs = len(CF_dataset.train_dict)
        self.max_sequence_len = max_sequence_len
        self.all_user_ids = torch.arange(0, self.num_users, dtype=torch.long)
        self.all_item_ids = torch.arange(1, self.num_items + 1, dtype=torch.long)
        self.seq_mat = torch.zeros((self.num_users, self.max_sequence_len), dtype=torch.long)
        self.train_pairs = CF_dataset.train_pairs
        
        for uid, seq in self.train_dict.items():
            seq_len = len(seq)
            seq = seq + 1   # item IDs start from 1
            if seq_len < self.max_sequence_len:
                seq = torch.concat([seq, torch.zeros(self.max_sequence_len - seq_len)])
            else:
                seq = seq[-self.max_sequence_len:]
            self.seq_mat[uid] = seq

    def __len__(self):
        return self.num_seqs
    
    def __getitem__(self, idx):
        return self.all_user_ids[idx], self.seq_mat[idx]


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
            user as keys, valid/test item as values
        """
        super(implicit_CF_dataset_test, self).__init__()
        self.user_num = num_users
        self.item_num = num_items
        self.user_list = torch.LongTensor([i for i in range(num_users)])

        self.inter_dict = deepcopy(inter_dict)


#################################################################################################################
# CTR datasets
#################################################################################################################

class CTRChunkLoader:
    def __init__(self, chunk_paths):
        self.chunk_paths = chunk_paths

    def __len__(self):
        return len(self.chunk_paths)

    def __iter__(self):
        for chunk_path in self.chunk_paths:
            yield torch.load(chunk_path, map_location="cpu", weights_only=True)


def _load_yaml_or_empty(path):
    if not os.path.exists(path):
        return {}
    config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    return config if config is not None else {}


def _infer_ctr_schema(train_csv):
    sample = pd.read_csv(train_csv, nrows=2048, low_memory=False)
    if "label" not in sample.columns:
        raise ValueError(f"Missing label column in {train_csv}")
    feature_names = [c for c in sample.columns if c != "label"]
    feature_types = {}
    for col in feature_names:
        if pd.api.types.is_float_dtype(sample[col].dtype):
            feature_types[col] = "numeric"
        else:
            feature_types[col] = "categorical"
    return feature_names, feature_types


def _build_ctr_split_cache(csv_path, split_cache_dir, batch_size, feature_names, feature_types):
    os.makedirs(split_cache_dir, exist_ok=True)
    categorical_max = {name: 0 for name, ftype in feature_types.items() if ftype == "categorical"}
    chunk_paths = []

    reader = pd.read_csv(csv_path, chunksize=batch_size, low_memory=False)
    for idx, chunk in enumerate(tqdm(reader, desc=f"Caching {os.path.basename(csv_path)}", leave=False)):
        chunk = chunk.fillna(0)
        batch = {"label": torch.from_numpy(chunk["label"].to_numpy(dtype=np.float32))}
        for name in feature_names:
            if feature_types[name] == "numeric":
                batch[name] = torch.from_numpy(chunk[name].to_numpy(dtype=np.float32))
            else:
                values = chunk[name].to_numpy(dtype=np.int64)
                batch[name] = torch.from_numpy(values)
                if values.size > 0:
                    categorical_max[name] = max(categorical_max[name], int(values.max()))
        chunk_path = os.path.join(split_cache_dir, f"chunk_{idx:06d}.pt")
        torch.save(batch, chunk_path)
        chunk_paths.append(chunk_path)

    meta = {"num_chunks": len(chunk_paths), "categorical_max": categorical_max}
    with open(os.path.join(split_cache_dir, "meta.yaml"), "w") as f:
        yaml.safe_dump(meta, f, sort_keys=True)
    return chunk_paths, categorical_max


def _prepare_ctr_split_loader(dataset_dir, split, batch_size, feature_names, feature_types):
    split_csv = os.path.join(dataset_dir, f"{split}.csv")
    split_cache_dir = os.path.join(dataset_dir, "cache", f"bs{batch_size}", split)
    meta_path = os.path.join(split_cache_dir, "meta.yaml")

    if os.path.exists(meta_path):
        meta = _load_yaml_or_empty(meta_path)
        chunk_paths = []
        for f_name in sorted(os.listdir(split_cache_dir)):
            if f_name.startswith("chunk_") and f_name.endswith(".pt"):
                chunk_paths.append(os.path.join(split_cache_dir, f_name))
        if meta.get("num_chunks", -1) == len(chunk_paths) and len(chunk_paths) > 0:
            return CTRChunkLoader(chunk_paths), meta.get("categorical_max", {})

    if os.path.exists(split_cache_dir):
        shutil.rmtree(split_cache_dir)
    chunk_paths, categorical_max = _build_ctr_split_cache(
        split_csv, split_cache_dir, batch_size, feature_names, feature_types
    )
    return CTRChunkLoader(chunk_paths), categorical_max


def get_ctr_dataset(args):
    dataset_dir = os.path.join(args.DATA_DIR, args.dataset)
    config_path = os.path.join(dataset_dir, "config.yaml")
    config = _load_yaml_or_empty(config_path)

    train_csv = os.path.join(dataset_dir, "train.csv")
    valid_csv = os.path.join(dataset_dir, "valid.csv")
    test_csv = os.path.join(dataset_dir, "test.csv")
    for p in [train_csv, valid_csv, test_csv]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing CTR file: {p}")

    if "feature_names" in config and "feature_types" in config:
        feature_names = config["feature_names"]
        feature_types = config["feature_types"]
    else:
        feature_names, feature_types = _infer_ctr_schema(train_csv)
        config["feature_names"] = feature_names
        config["feature_types"] = feature_types

    train_loader, train_max = _prepare_ctr_split_loader(dataset_dir, "train", args.batch_size, feature_names, feature_types)
    valid_loader, valid_max = _prepare_ctr_split_loader(dataset_dir, "valid", args.batch_size, feature_names, feature_types)
    test_loader, test_max = _prepare_ctr_split_loader(dataset_dir, "test", args.batch_size, feature_names, feature_types)

    # Build feature_stastic from cache metadata, and keep config.yaml synced.
    feature_stastic = config.get("feature_stastic", {"label": 2})
    for name, ftype in feature_types.items():
        if ftype == "categorical":
            cache_max = max(train_max.get(name, 0), valid_max.get(name, 0), test_max.get(name, 0))
            feature_stastic[name] = cache_max + 1
        else:
            feature_stastic[name] = 1
    feature_stastic["label"] = 2
    config["feature_names"] = feature_names
    config["feature_types"] = feature_types
    config["feature_stastic"] = feature_stastic
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=True)

    return train_loader, valid_loader, test_loader, feature_stastic, feature_types
