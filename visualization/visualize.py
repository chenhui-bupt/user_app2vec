# -*- coding: utf-8 -*-
import sys, os
sys.path.append('..')
import tensorflow as tf
import shutil
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from recommendation.data_process import get_node2id
from recommendation.data_generate import get_embeddings


def visualize(app2vec, applist, output_path):
    meta_file = "app_metadata.tsv"
    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
        file_metadata.write(b'\n'.join(map(lambda app: "{0}".format(app).encode('utf-8'), applist)))

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(app2vec, trainable=False, name='app2vec_metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'app2vec_metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path, 'app2vec_metadata.ckpt'))
    print('referred to https://eliyar.biz/using-pre-trained-gensim-word2vector-in-a-keras-model-and-visualizing/')
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))


if __name__ == "__main__":
    embeddings_file = ''
    if sys.argv[1] == 'deepwalk':
        embeddings_file = 'deepwalk.embeddings'
    elif sys.argv[1] == 'hin2vec':
        embeddings_file = 'node_vectors.txt'
    node2id = get_node2id()
    app_embeddings = get_embeddings(embeddings_file, node2id)[25413:25467]
    app_embeddings /= np.sqrt(np.sum(np.square(app_embeddings), axis=1, keepdims=True))
    applist = [0] * 54
    for node, val in node2id.items():
        if 25413 <= val[0] < 25467:
            applist[val[0] - 25413] = node
    outpath = './output/%s' % sys.argv[1]
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    visualize(app_embeddings, applist, outpath)
