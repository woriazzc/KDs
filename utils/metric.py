import numpy as np


def get_labels(sorted_list, target_items, num_items):
    vis = np.zeros(num_items)
    for i in target_items:
        vis[i] = 1
    labels = np.zeros(len(sorted_list))
    for idx, i in enumerate(sorted_list):
        labels[idx] = vis[i]
    return labels

def Precision(labels, num_targets):
    precision = labels.mean(-1)
    return precision.mean()

def Recall(labels, num_targets):
    num_targets = np.maximum(num_targets, 1)
    recall = labels.sum(-1) / num_targets
    return recall.mean()

def NDCG(labels, num_targets):
    num_targets = np.maximum(num_targets, 1)
    K = labels.shape[1]
    test_matrix = np.zeros((len(labels), K))
    for i in range(len(labels)):
        length = min(num_targets[i], K)
        test_matrix[i, :length] = 1
    denom = np.log2(np.arange(2, K + 2)).reshape(1, -1)
    dcg = np.sum(labels / denom, axis=-1)
    idcg = np.sum(test_matrix / denom, axis=-1)
    ndcg = dcg / idcg
    return ndcg.mean()
