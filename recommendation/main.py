# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf
from models.dnn import DNN
from data_generate import *
from data_process import get_node2id


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


if __name__ == '__main__':
    if sys.argv[1] == 'deepwalk':
        embeddings_file = 'deepwalk.embeddings'
    elif sys.argv[1] == 'hin2vec':
        embeddings_file = 'node_vectors.txt'
    node2id = get_node2id()
    node_embeddings = get_embeddings(embeddings_file, node2id)
    train_dataset, test_dataset = train_test_split('./data/all_data.csv', train_size=0.7)
    model = DNN(config=config, batch_size=2048, node_embeddings=node_embeddings, optimizer='adam', learning_rate=0.001,
                epoch_num=5)
    model.train(train_dataset=train_dataset, test_dataset=test_dataset)
