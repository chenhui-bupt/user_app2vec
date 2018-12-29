# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd


def train_test_split(file, train_size):
    train_dataset = []
    test_dataset = []
    with open(file, errors='ignore') as f:
        f.readline()
        for line in f:
            splits = line.split(',')
            if random.random() < train_size:
                train_dataset.append(list(map(float, splits)))
            else:
                test_dataset.append(list(map(float, splits)))
    return train_dataset, test_dataset


def get_embeddings(file_name, node2id=None):  # 每个节点的embedding向量，给dnn的lookup用
    """
    embeddings的节点有的存成id, 有的是名称需要node2id
    :param file_name: embeddings file
    :param node2id: node:[id, type, count]
    :return:
    """
    embeddings_file = os.path.join("../network_embedding/embeddings_output/", file_name)
    with open(embeddings_file, 'r') as f:
        shape = tuple(map(int, f.readline().split()))
        print('node: %s, embedding_size: %s' % shape)
        embeddings = np.zeros(shape)
        for line in f:
            splits = line.split()
            if node2id:
                embeddings[node2id[splits[0]][0]] = list(map(float, splits[1:]))
            else:
                embeddings[int(splits[0])] = list(map(float, splits[1:]))
    return embeddings


def random_embedding(nodesize, embedding_dim):
    """

    :param nodesize:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (nodesize, embedding_dim))  # 均匀初始化
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat

def to_categorical(y, num_class=None):  # onehot or multihot
    '''

    :param y: type=numpy.array, y.shape=(None,) or (None, None), 也就是onehot或者multihot
    :param num_class: 类别数
    :return:
    '''
    if not num_class:
        num_class = np.max(y) + 1
    out = np.zeros([len(y), num_class])
    for i in range(len(y)):
        out[i, y[i]] = 1
    return out

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
        labels.append(int(d[-1]))
        if len(feats) == batch_size:  # 当其够一个batch时，就yield出去
            yield np.array(feats), to_categorical(np.array(labels), num_class=2)
            feats, labels = [], []
    if len(feats) != 0:  # 最后不能整除的余数部分，也要yield
        yield np.array(feats), to_categorical(np.array(labels), num_class=2)
