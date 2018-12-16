# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd


def get_embeddings(file_name, node2id=None):  # 每个节点的embedding向量，给dnn的lookup用
    embeddings = {}
    embeddings_file = os.path.join("../network_embedding/embeddings_output/", file_name)
    with open(embeddings_file, 'r') as f:
        for line in f:
            splits = line.split()
            if len(splits) == 2:
                print('node: %s, embedding_size: %s' % tuple(splits))
                continue
            if node2id:
                embeddings[node2id[splits[0]]] = list(map(float, splits[1:]))
            else:
                embeddings[int(splits[0])] = list(map(float, splits[1:]))
    return embeddings


def random_embedding(node2id, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(node2id), embedding_dim))  # 均匀初始化
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def batch_yield(data, batch_size, shuffle=False):
    """

    :param data: matrix, [total_samples, num_features]
    :param batch_size:
    :param shuffle:
    :return: batches of [batch_size, (feat1, feat2, ..., featn, label)]
    """
    if shuffle:  # 对数据进行shuffle
        random.shuffle(data)
    feats, labels = [], []
    for d in data:
        feats.append(d[:-1])
        labels.append(d[-1])
        if len(feats) == batch_size:  # 当其够一个batch时，就yield出去
            yield feats, labels
            feats, labels = [], []
    if len(feats) != 0:  # 最后不能整除的余数部分，也要yield
        yield feats, labels
