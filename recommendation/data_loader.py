# -*- coding:utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import random
import pickle


def get_node2id(file):
    if not file:
        file = '../resources/node2id.pkl'
    node2id = pickle.load(open(file, 'rb'))
    print('load node2id from %s, it has %s nodes' % (file, len(node2id)))
    return node2id


def to_categorical(y, num_class=None):  # onehot or multihot
    '''

    :param y: y.shape=(None,) or (None, None), 也就是onehot或者multihot
    :param num_class: 类别数
    :return:
    '''
    if not num_class:
        num_class = np.max(y) + 1
    out = np.zeros([len(y), num_class])
    for i in range(len(y)):
        out[i, y[i]] = 1
    return out


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

    :param data:
    :param batch_size:
    :param shuffle:
    :return:
    """
    if shuffle:  # 对数据进行shuffle
        random.shuffle(data)
    batch = []
    for d in data:
        batch.append(d)
        if len(batch) == batch_size:  # 当其够一个batch时，就yield出去
            yield batch
            batch = []
    if len(batch) != 0:  # 最后不能整除的余数部分，也要yield
        yield batch


