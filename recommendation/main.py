# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from models.dnn import DNN
from data_generate import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


node_embeddings = get_embeddings('deepwalk.embeddings')
train_dataset, test_dataset = train_test_split('./data/all_data.csv', train_size=0.7)
model = DNN(config=config, batch_size=2048, node_embeddings=node_embeddings, optimizer='adam', learning_rate=0.001, epoch_num=5)
model.train(train_dataset=train_dataset, test_dataset=test_dataset)



