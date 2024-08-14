import os
import pickle
import numpy as np

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


def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape), dtype=torch.float32)


def sum_emb_out_dim(feature_map, emb_dim, feature_source=[]):
        if type(feature_source) != list:
            feature_source = [feature_source]
        total_dim = 0
        for feature, feature_spec in feature_map.features.items():
            if feature_spec["type"] == "meta":
                continue
            if len(feature_source) == 0 or feature_spec.get("source") in feature_source:
                total_dim += feature_spec.get("emb_output_dim",
                                              feature_spec.get("embedding_dim", 
                                                               emb_dim))
        return total_dim
