import os
import math
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
import copy
import tensorflow as tf
import tensorflow_probability as tfp


class VI:

    def __init__(self, goal_range, input_range, s_1, s_2, name):

        self.s_1 = s_1
        self.s_2 = s_2
        self.name = name

        self.input_range = input_range
        self.goal_range = goal_range

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.create_network()

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True

            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

    def create_network(self):

        self.data_input = tf.placeholder(tf.float32, shape=[None, self.input_range])
        self.labels = tf.placeholder(tf.int32, shape=[None, ])
        self.num_data = tf.placeholder(tf.float32, shape=[1])

        with tf.device('gpu:0/'):
            self.model = tf.keras.Sequential([
                tfp.layers.DenseReparameterization(32, activation=tf.nn.relu),
                tfp.layers.DenseReparameterization(32, activation=tf.nn.relu),
                tfp.layers.DenseReparameterization(self.goal_range),
            ])
            self.logits = self.model(self.data_input)
            self.output = tf.nn.softmax(self.logits)

            neg_log_likelihood = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.labels, logits=self.logits)
            self.log_loss = tf.reduce_mean(neg_log_likelihood)
            self.kl = 1. * sum(self.model.losses) / self.num_data
            self.loss = self.log_loss + self.kl
            self.train_op = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(self.loss)

    def run(self, labels, feature):
        with self.graph.as_default():
            res = self.sess.run(self.output, feed_dict={self.data_input: feature})
            res = np.take(res, labels)
        return res

    def update(self, labels, feature):
        num_data = labels.shape[0]
        with self.graph.as_default():
            _, loss = \
                self.sess.run([self.train_op, self.log_loss],
                              feed_dict={self.data_input: feature,
                                         self.labels: labels,
                                         self.num_data : np.array([num_data])})
            #if self.name == 'p' and self.s_2 == 1:
            #    print(loss)


class fast_VI:

    def __init__(self, goal_range, input_range, n_agent, name):
        self.n_agent = n_agent
        self.name = name

        self.input_range = input_range
        self.goal_range = goal_range

        self.total = n_agent * (n_agent - 1)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.create_network()

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True

            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

    def create_network(self):
        self.data_input = tf.placeholder(tf.float32, shape=[None, self.total, self.input_range])
        self.labels = tf.placeholder(tf.int32, shape=[None, self.total])
        self.num_data = tf.placeholder(tf.float32, shape=[1])
        with tf.device('gpu:0/'):
            output_list = []
            self.log_loss = []
            self.train_op = []
            for i in range(2):
                with tf.name_scope('%d' % i):
                    inputs = tf.keras.Input(shape=(None, self.input_range))
                    x_1 = tfp.layers.DenseReparameterization(32, activation=tf.nn.relu, name='01')(inputs)
                    x_2 = tfp.layers.DenseReparameterization(32, activation=tf.nn.relu, name='02')(x_1)
                    outputs = tfp.layers.DenseReparameterization(self.goal_range, name='03')(x_2)
                    t_model = tf.keras.Model(inputs=inputs, outputs=outputs)
                    t_logits = t_model(self.data_input[:, i])
                    t_output = tf.nn.softmax(t_logits)
                    output_list.append(tf.expand_dims(t_output, 1))

                    neg_log_likelihood = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.labels[:, i], logits=t_logits)
                    t_log_loss = tf.reduce_mean(neg_log_likelihood)
                    self.log_loss.append(t_log_loss)
                    t_kl = 1. * sum(t_model.losses) / self.num_data
                    t_loss = t_log_loss + t_kl
                    t_train_op = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(t_loss)
                    self.train_op.append(t_train_op)
            self.output = tf.concat(output_list, axis=1)

    def run(self, labels, feature):
        with self.graph.as_default():
            t_feature = np.transpose(feature, (1, 0, 2))
            t_labels = np.transpose(labels, (1, 0))
            res = self.sess.run(self.output, feed_dict={self.data_input: t_feature})
            shape = res.shape
            res = np.reshape(np.take(np.reshape(res, (-1, shape[2])), np.reshape(t_labels, (-1))), (shape[0], shape[1]))
            res = np.transpose(res, (1, 0))
        return res

    def update(self, labels, feature):
        with self.graph.as_default():
            t_feature = np.transpose(feature, (1, 0, 2))
            t_labels = np.transpose(labels, (1, 0))
            num_data = t_labels.shape[0]
            output = \
                self.sess.run([self.train_op[1]],
                              feed_dict={self.data_input: t_feature,
                                         self.labels: t_labels,
                                         self.num_data: np.array([num_data])})
            #if self.name == 'p':
            #    print(output[self.total])


class Test_vi:

    def __init__(self, n_agent, goal_range, size, a_size, input_range_p, input_range_t, appro_T):

        self.n_agent = n_agent
        self.goal_range = goal_range
        self.size = size
        self.a_size = a_size
        self.appro_T = appro_T
        self.input_range_p = input_range_p
        self.input_range_t = input_range_t

        self.eye = np.eye(self.size)
        self.eye_action = np.eye(self.a_size)

        self.batch_size = 2048
        self.collect = 0
        self.collect_label = []
        self.collect_p = []
        self.collect_t = []

        self.model_p = []
        self.model_t = []
        for i in range(self.n_agent):
            for j in range(self.n_agent):
                if i != j:
                    self.model_p.append(VI(self.goal_range, self.input_range_p, j, i, 'p'))
                    self.model_t.append(VI(self.goal_range, self.input_range_t, j, i, 't'))

    def make(self, goal, state):

        v_del_i = []
        v_s_t_i = []
        v_a_t_i = []
        for i in range(self.n_agent):
            v_del_i.append(goal[i])
            v_s_t_i.append(self.eye[state[i][0]])
            v_a_t_i.append(self.eye_action[state[i][1]])

        p_num_label = []
        p_num_x_p = []
        p_num_x_t = []
        for i in range(self.n_agent):
            y = v_del_i[i]
            x_p = np.concatenate([v_s_t_i[i], v_a_t_i[i]], axis=0)
            for j in range(self.n_agent):
                if i != j:
                    x_t = np.concatenate([x_p, v_s_t_i[j], v_a_t_i[j]], axis=0)
                    p_num_label.append(y)
                    p_num_x_p.append(x_p)
                    p_num_x_t.append(x_t)
        return p_num_label, p_num_x_p, p_num_x_t

    def update(self):

        labels = np.concatenate(self.collect_label, axis=1)
        feature_p = np.concatenate(self.collect_p, axis=1)
        feature_t = np.concatenate(self.collect_t, axis=1)

        t_stamp = 0
        for i in range(self.n_agent):
            if i == 0:
                t_stamp += 1
                continue
            for j in range(self.n_agent):
                if i != j:
                    self.model_p[t_stamp].update(labels[t_stamp], feature_p[t_stamp])
                    self.model_t[t_stamp].update(labels[t_stamp], feature_t[t_stamp])
                    t_stamp += 1

        self.collect = 0
        self.collect_label = []
        self.collect_p = []
        self.collect_t = []

    def output(self, label, x_p, x_t):

        self.collect_label.append(label)
        self.collect_p.append(x_p)
        self.collect_t.append(x_t)

        num_data = label.shape[1]
        self.collect += num_data
        if self.collect >= self.batch_size:
            self.update()

        coor_rewards = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
        coor_p = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
        coor_t = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')

        t_stamp = 0
        for i in range(self.n_agent):
            if i == 0:
                t_stamp += 1
                continue
            for j in range(self.n_agent):
                if i != j:
                    prob_p = self.model_p[t_stamp].run(label[t_stamp], x_p[t_stamp])
                    prob_t = self.model_t[t_stamp].run(label[t_stamp], x_t[t_stamp])
                    t_stamp += 1
                    coor_p[j][i] = prob_p
                    coor_t[j][i] = prob_t
                    coor_rewards[j][i] = 1. - prob_p / np.maximum(prob_t, self.appro_T)

        return coor_rewards, coor_p, coor_t

    def show(self, e):
        pass


class fast_Test_vi:

    def __init__(self, n_agent, goal_range, size, a_size, input_range_p, input_range_t, appro_T):

        self.n_agent = n_agent
        self.goal_range = goal_range
        self.size = size
        self.a_size = a_size
        self.appro_T = appro_T
        self.input_range_p = input_range_p
        self.input_range_t = input_range_t

        self.eye = np.eye(self.size)
        self.eye_action = np.eye(self.a_size)

        self.batch_size = 2048
        self.collect = 0
        self.collect_label = []
        self.collect_p = []
        self.collect_t = []

        '''
        self.model_p = []
        self.model_t = []
        for i in range(self.n_agent):
            for j in range(self.n_agent):
                if i != j:
                    self.model_p.append(VI(self.goal_range, self.input_range_p, j, i, 'p'))
                    self.model_t.append(VI(self.goal_range, self.input_range_t, j, i, 't'))
        '''

        self.model_p = fast_VI(self.goal_range, self.input_range_p, self.n_agent, 'p')
        self.model_t = fast_VI(self.goal_range, self.input_range_t, self.n_agent, 't')

    def make(self, goal, state):

        v_del_i = []
        v_s_t_i = []
        v_a_t_i = []
        for i in range(self.n_agent):
            v_del_i.append(goal[i])
            v_s_t_i.append(self.eye[state[i][0]])
            v_a_t_i.append(self.eye_action[state[i][1]])

        p_num_label = []
        p_num_x_p = []
        p_num_x_t = []
        for i in range(self.n_agent):
            y = v_del_i[i]
            x_p = np.concatenate([v_s_t_i[i], v_a_t_i[i]], axis=0)
            for j in range(self.n_agent):
                if i != j:
                    x_t = np.concatenate([x_p, v_s_t_i[j], v_a_t_i[j]], axis=0)
                    p_num_label.append(y)
                    p_num_x_p.append(x_p)
                    p_num_x_t.append(x_t)
        return p_num_label, p_num_x_p, p_num_x_t

    def update(self):

        labels = np.concatenate(self.collect_label, axis=1)
        feature_p = np.concatenate(self.collect_p, axis=1)
        feature_t = np.concatenate(self.collect_t, axis=1)

        '''
        t_stamp = 0
        for i in range(self.n_agent):
            if i == 0:
                t_stamp += 1
                continue
            for j in range(self.n_agent):
                if i != j:
                    self.model_p[t_stamp].update(labels[t_stamp], feature_p[t_stamp])
                    self.model_t[t_stamp].update(labels[t_stamp], feature_t[t_stamp])
                    t_stamp += 1
        '''

        self.model_p.update(labels, feature_p)
        self.model_t.update(labels, feature_t)

        self.collect = 0
        self.collect_label = []
        self.collect_p = []
        self.collect_t = []

    def output(self, label, x_p, x_t):

        self.collect_label.append(label)
        self.collect_p.append(x_p)
        self.collect_t.append(x_t)

        num_data = label.shape[1]
        self.collect += num_data
        if self.collect >= self.batch_size:
            self.update()

        coor_rewards = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
        coor_p = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
        coor_t = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')

        '''
        t_stamp = 0
        for i in range(self.n_agent):
            if i == 0:
                t_stamp += 1
                continue
            for j in range(self.n_agent):
                if i != j:
                    prob_p = self.model_p[t_stamp].run(label[t_stamp], x_p[t_stamp])
                    prob_t = self.model_t[t_stamp].run(label[t_stamp], x_t[t_stamp])
                    t_stamp += 1
                    coor_p[j][i] = prob_p
                    coor_t[j][i] = prob_t
                    coor_rewards[j][i] = 1. - prob_p / np.maximum(prob_t, self.appro_T)
        '''

        prob_p = self.model_p.run(label, x_p)
        prob_t = self.model_t.run(label, x_t)

        t_stamp = 0
        for i in range(self.n_agent):
            if i == 0:
                continue
            for j in range(self.n_agent):
                if i != j:
                    coor_p[j][i] = prob_p[t_stamp]
                    coor_t[j][i] = prob_t[t_stamp]
                    coor_rewards[j][i] = 1. - prob_p[t_stamp] / np.maximum(prob_t[t_stamp], self.appro_T)
                    t_stamp += 1

        return coor_rewards, coor_p, coor_t

    def show(self, e):
        pass


def test_vi(test, goals, state_1, state_2):
    C_label = []
    C_x_p = []
    C_x_t = []
    for i, goal in enumerate(goals):
        state = (state_1[i], state_2[i])
        label, x_p, x_t = test.make(goal, state)
        C_label.append(label)
        C_x_p.append(x_p)
        C_x_t.append(x_t)
    C_label = np.transpose(np.array(C_label), (1, 0))
    C_x_p = np.transpose(np.array(C_x_p), (1, 0, 2))
    C_x_t = np.transpose(np.array(C_x_t), (1, 0, 2))
    coor_output = test.output(C_label, C_x_p, C_x_t)
    return coor_output


def print_coor(test, goals, state_1, state_2):
    coor_output = test_vi(test, goals, state_1, state_2)
    coor_rewards, coor_p, coor_t = coor_output
    print('------------------')
    print('coor_rewards')
    print(coor_rewards[0][1])
    print('coor_p')
    print(coor_p[0][1])
    print('coor_t')
    print(coor_t[0][1])


def main():
    size = 10
    a_size = 6
    goal_size = 55
    test = Test_vi(2, goal_size, size, a_size, 16, 32, 1.)
    for i in range(10000):
        fix_goal = [2, 2]
        fix_s_t_1 = np.array([9, 2])
        fix_s_t_2 = np.array([5, 2])
        goals = []
        state_1 = []
        state_2 = []
        tim = 2048
        for j in range(tim):
            s_t_1 = [np.random.randint(0, size), np.random.randint(0, a_size)]
            s_t_2 = [np.random.randint(0, size), np.random.randint(0, a_size)]
            if s_t_1[0] == 9:
                s_t_1[1] = 2
            else:
                s_t_1[1] = 2
            goal = [s_t_2[1], s_t_2[1]] if s_t_1[0] != 9 or (s_t_2 != fix_s_t_2).any() else [6, 6]
            goals.append(goal)
            state_1.append(s_t_1)
            state_2.append(s_t_2)
        coor_output = test_vi(test, goals, state_1, state_2)

        if i % 500 == 0:
            print('------------------')
            print(i)
            print_coor(test, [[0, 0]], [[4, 0]], [[5, 0]])
            print_coor(test, [[2, 2]], [[6, 3]], [[5, 2]])
            print_coor(test, [[0, 0]], [[9, 2]], [[4, 0]])
            print_coor(test, [[0, 0]], [[9, 2]], [[5, 0]])
            print_coor(test, [[6, 6]], [[9, 2]], [[5, 2]])


if __name__ == '__main__':
    main()
