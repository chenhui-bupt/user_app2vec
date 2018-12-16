import os
import sys
import numpy as np
import time
import tensorflow as tf
from recommendation.data_generate import batch_yield

class DNN(object):
    def __init__(self, batch_size, node_embedding, config, optimizer):
        self.batch_size = batch_size
        self.node_embedding = node_embedding
        self.update_embedding = False
        self.num_classes = 1
        self.clip = False
        self.clip_value = 5.0
        self.optimizer = optimizer
        self.save_path = '../output/dnn/'

    def build_graph(self):
        self.add_placeholders()
        self.lookup_input()
        self.fc_layer()
        self.loss()
        self.train_step()
        self.variables_init_op()

    def add_placeholders(self):
        self.nodepair_input = tf.placeholder(tf.int32, [None, 2], name='nodepair_input')  # user-app pair
        self.X_input = tf.placeholder(tf.float32, [None, None], name='X_input')
        self.y_input = tf.placeholder(tf.int32, [None, ], name='y_input')  # label

    def lookup_input(self):
        with tf.variable_scope('node_embedding'):
            _node_embedding = tf.get_variable('node_embedding', shape=self.node_embedding.shape,
                                                  initializer=tf.constant_initializer(self.node_embedding),
                                                  trainable=self.update_embedding)
        # lookup
        self.nodepair_embedding = tf.nn.embedding_lookup(
            params=_node_embedding,
            ids=self.nodepair_input,
            name='nodepair_embedding'
        )

    def fc_layer(self):
        with tf.name_scope('fc1'):
            inputs = tf.concat([self.nodepair_embedding, self.X_input], axis=-1)
            self.input_size = inputs.shape[0]
            self.w1 = tf.get_variable(name='w1',
                                      shape=[self.input_size, 512],
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      dtype=tf.float32)
            self.b1 = tf.get_variable(name='b1',
                                      shape=[512],
                                      initializer=tf.zeros_initializer(),
                                      dtype=tf.float32)
            self.l1 = tf.nn.relu(tf.matmul(inputs, self.w1) + self.b1)

        with tf.name_scope('fc2'):
            self.w2 = tf.get_variable(name='w2',
                                      shape=[512, 256],
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      dtype=tf.float32)
            self.b2 = tf.get_variable(name='b2',
                                      shape=[256],
                                      initializer=tf.zeros_initializer(),
                                      dtype=tf.float32)
            self.l2 = tf.nn.relu(tf.matmul(self.l1, self.w2) + self.b2)

        with tf.name_scope('fc3'):
            self.w3 = tf.get_variable(name='w3', shape=[256, self.num_classes],
                                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.b3 = tf.get_variable(name='b3', shape=[self.num_classes],
                                      initializer=tf.zeros_initializer(), dtype=tf.float32)
            self.logits = tf.matmul(self.l2, self.w3) + self.b3

    def loss(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_input, logits=self.logits))

    def train_step(self, loss, learning_rate):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)  # False
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif self.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        elif self.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        elif self.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif self.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        if self.clip:
            grads_and_vars = optimizer.compute_gradients(loss)
            grads_and_vars_clipped = [[tf.clip_by_value(g, -self.clip_value, self.clip_value), v] for g, v in grads_and_vars]
            self.train_op = optimizer.apply_gradients(grads_and_vars_clipped, global_step=self.global_step)
        else:
            self.train_op = optimizer.minimize(loss, global_step=self.global_step)

    def variables_init_op(self):
        self.init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    def add_summary(self, sess):
        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.save_path + 'summary/', sess.graph)  # 保存图像

    def train(self, train_dataset, test_dataset):
        """

        :return:
        """
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            for epoch in range(self.epoch_num):
                num_batches = (len(train_dataset) + self.batch_size - 1)//self.batch_size
                start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                batches = batch_yield(train_dataset, self.batch_size, shuffle=True)
                for i, (X_inputs, labels) in enumerate(batches):
                    sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')  # 回到行首
                    feed_dict = {
                        self.nodepair_input: X_inputs[:, :2],
                        self.X_input: X_inputs[:, 2:],
                        self.y_input: labels
                    }
                    _train_op, _loss, _merged, _step = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                                feed_dict=feed_dict)
                    self.file_writer.add_summary(_merged, _step)  # add summary
                    if i + 1 == num_batches:  # one epoch
                        saver.save(sess, self.save_path + '/model', _step)

                # TODO validation/test
                validate_one_epoch(test_dataset, sess)
                self.evaluate()








