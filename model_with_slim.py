__author__ = 'Zhenan Ye'

import tensorflow as tf

import numpy as np
import os
import data_utils
from flip_gradient import flip_gradient
import argparse


slim = tf.contrib.slim

# # model params for UniMiB
# channels = 3
# cut_channels = channels
# window_length = 151
# num_classes = 17
# num_domain_labels = 2
# # feature_extractor 27, residual_feat_extractor 16,26
# final_length = 26
# kernel_sizes = [[2, 1], [3, 1]]
# strides = [[2, 1], [3, 1]]

# model params for Realistics
channels = 117
window_length = 300
num_classes = 33
num_domain_labels = 2
final_length = 50
cut_channels = 20
kernel_sizes = [[2, 2], [3, 3]]
strides = [[2, 2], [3, 3]]

# model params for Realistics
# channels = 113
# window_length = 24
# num_classes = 17
# num_domain_labels = 2
# final_length = 6
# cut_channels = channels
# kernel_sizes = [[2, 1], [2, 1]]
# strides = [[2, 1], [2, 1]]

# train params
batch_size = 100
num_filter = 64
num_units_lstm = 128
num_steps = 10000
summary_step = 300
start_learning_rate = 0.00007
lr_decay_steps = 200
lr_decay_rate = 0.9
# start_learning_rate = 0.001
# lr_decay_steps = 500
# lr_decay_rate = 0.95

def create_placeholder():
    X = tf.placeholder(shape=[None, window_length, channels], dtype=tf.float32)
    Y = tf.placeholder(shape=[None, num_classes], dtype=tf.float32)
    D_Y = tf.placeholder(shape=[None, num_domain_labels], dtype=tf.float32)
    prok = tf.placeholder(dtype=tf.float32)
    is_train = tf.placeholder(dtype=tf.bool)
    sigma = tf.placeholder(dtype=tf.float32)
    is_training = tf.placeholder(dtype=tf.bool)
    return X, Y, D_Y, prok, is_train, sigma, is_training

def resnet_block(net, resnet_filters):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
        net_in = net
        net = slim.conv2d(
            net,
            resnet_filters,
            stride=1,
            normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu)
        net = slim.conv2d(
            net,
            resnet_filters,
            stride=1,
            normalizer_fn=slim.batch_norm,
            activation_fn=None)
        if net_in.get_shape()[-1] != resnet_filters:
            net_in = slim.conv2d(
                net_in,
                resnet_filters,
                stride=1,
                normalizer_fn=slim.batch_norm,
                activation_fn=None)

        return tf.nn.relu(net + net_in)

def label_predictor(features, prok):
    with tf.variable_scope('label_predictor'):
        # flatten_features = slim.flatten(features)
        # with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
        #     f_net1 = slim.fully_connected(flatten_features, 1024, scope='full1')
        #     f_net2 = slim.fully_connected(f_net1, 128, scope='full2')
        #     logits = slim.fully_connected(
        #         f_net2, num_classes, activation_fn=None, scope='fc_logits')
        features = tf.transpose(features, [1, 0, 2, 3])  # (135, batch_size, 3, 64)

        feat_flat = tf.reshape(features, [-1, cut_channels * num_filter])

        lstm_inputs = tf.split(feat_flat, num_or_size_splits=final_length, axis=0)
        with tf.variable_scope("lstm_layers"):
            lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(num_units_lstm, forget_bias=1.0, state_is_tuple=True)
            lstm_cell_1 = tf.contrib.rnn.DropoutWrapper(lstm_cell_1, input_keep_prob=prok, output_keep_prob=prok)
            lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units_lstm, forget_bias=1.0, state_is_tuple=True)
            cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
            outputs, _ = tf.contrib.rnn.static_rnn(cells, lstm_inputs, dtype=tf.float32)
            logits = slim.fully_connected(
                outputs[-1], num_classes, activation_fn=None, scope='fc_logits')

    return logits

def domain_predictor(features):
    with tf.variable_scope('domain_predictor'):
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
            f_net1 = slim.fully_connected(features, 1024, scope='full1')
            f_net2 = slim.fully_connected(f_net1, 128, scope='full2')
            logits = slim.fully_connected(f_net2, 2, activation_fn=None, scope='fc_logits')
    return logits

def batch_norm_params(is_training, batch_norm_decay):
    return {'is_training': is_training,
            'decay': batch_norm_decay,
            'epsilon': 0.001}

def loss(label_logits, domain_logits, labels, D_Y):
    # label loss
    label_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=label_logits, labels=labels))

    # domain loss
    domain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        # logits=tf.clip_by_value(domain_logits, 1e-10, 1.0), labels=D_Y))
        logits=domain_logits, labels=D_Y))

    total_loss = label_loss + domain_loss

    tf.summary.scalar("label_loss", label_loss)
    tf.summary.scalar('domain_loss', domain_loss)
    tf.summary.scalar('total_loss', total_loss)


    return label_loss, domain_loss, total_loss

def get_predictions(label_logits, domain_logits, labels, D_Y):
    label_prediction = tf.equal(tf.argmax(label_logits, 1), tf.argmax(labels, 1))
    domain_prediction = tf.equal(tf.argmax(domain_logits, 1), tf.argmax(D_Y, 1))

    return label_prediction, domain_prediction

def accuracy(label_logits, domain_logits, labels, D_Y):
    label_prediction = tf.equal(tf.argmax(label_logits, 1), tf.argmax(labels, 1))
    label_acc = tf.reduce_mean(tf.cast(label_prediction, tf.float32))
    tf.summary.scalar("label_accuracy", label_acc)

    domain_prediction = tf.equal(tf.argmax(domain_logits, 1), tf.argmax(D_Y, 1))
    domain_acc = tf.reduce_mean(tf.cast(domain_prediction, tf.float32))
    tf.summary.scalar('domain_accuracy', domain_acc)

    return label_acc, domain_acc


def create_model(x, is_training, is_train, sigma, Y, prok):
    with tf.variable_scope('dann'):
        X = tf.expand_dims(x, axis=3)  # batch_size, seq_length, channels, 1
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            # activation_fn=tf.nn.relu,
                            normalizer_params=batch_norm_params(is_training, 0.9),
                            weights_initializer=tf.random_normal_initializer(stddev=0.02),
                            weights_regularizer=tf.contrib.layers.l2_regularizer(1e-5)):
            with slim.arg_scope([slim.conv2d], kernel_size=[3, 3]):
                with slim.arg_scope([slim.max_pool2d], padding='SAME'):
                    net1 = resnet_block(X, 32)
                    net2 = resnet_block(net1, 32)

                    pool_net1 = slim.max_pool2d(net2, kernel_sizes[0], stride=strides[0], scope='pool1')

                    net3 = resnet_block(pool_net1, 64)
                    net4 = resnet_block(net3, 64)

                    all_features = slim.max_pool2d(net4, kernel_sizes[1], stride=strides[1], scope='pool2')

            # source_features = tf.slice(all_features, [0, 0, 0, 0], [batch_size/2, -1, -1, -1])
            classfier_features = tf.cond(is_train, lambda: tf.slice(all_features, [0, 0, 0, 0], [batch_size/2, -1, -1, -1]),
                                         lambda: all_features)
            all_labels = Y
            # source_labels = tf.slice(all_labels, [0, 0], [batch_size/2, -1])
            labels = tf.cond(is_train, lambda: tf.slice(all_labels, [0, 0], [batch_size/2, -1]), lambda: all_labels)


            label_logits = label_predictor(classfier_features, prok)

            all_features_flat = tf.reshape(all_features, [-1, final_length * cut_channels * num_filter])
            feat = flip_gradient(all_features_flat, sigma)

            domain_logits = domain_predictor(feat)

        return label_logits, domain_logits, labels

def batch_generator(data, batch_size):
    train_size = data[0].shape[0]
    pos = 0
    while True:
        scale = pos + batch_size
        if scale > train_size:
            start = pos
            a = scale - train_size
            pos = a
            yield [np.concatenate((d[start:], d[0:a])) for d in data]
        else:
            start = pos
            pos = scale
            yield [d[start:scale] for d in data]


def get_checkpoint(step):
    checkpoint = 1000
    if step > 3000:
        checkpoint = 500
    if step > 5000:
        checkpoint = 200
    return checkpoint

def train(training_mode, version, gpu, data_path):
    with tf.Graph().as_default():
        # create place_holder
        X, Y, D_Y, prok, is_train, sigma, is_training = create_placeholder()
        # create model
        label_logits, domain_logits, labels = create_model(X, is_training, is_train, sigma, Y, prok)

        # define loss
        label_loss, domain_loss, total_loss = loss(label_logits, domain_logits, labels, D_Y)

        # define accuracy
        label_acc, domain_acc = accuracy(label_logits, domain_logits, labels, D_Y)

        # train operations

        global_step = slim.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(
            start_learning_rate,
            global_step,
            decay_steps=lr_decay_steps,
            decay_rate=lr_decay_rate,
            staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        var_list, update_ops = data_utils.get_vars_and_update_ops('dann')
        dann_train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            update_ops=update_ops,
            variables_to_train=var_list,
            clip_gradient_norm=5,
            summarize_gradients=False)

        label_var_list, label_update_ops = data_utils.get_vars_and_update_ops('dann')
        label_train_op = slim.learning.create_train_op(
            label_loss,
            optimizer,
            update_ops=label_update_ops,
            variables_to_train=label_var_list,
            clip_gradient_norm=5,
            summarize_gradients=False)

        source_train, source_test, target_train, target_test = data_utils.load_data(data_path)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = os.path.join('logs', version)

        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        # saver = tf.train.Saver()
        saver = tf.train.Saver(slim.get_variables(), max_to_keep=50)
        train_saved_dir = os.path.join('saved_checkpoint', version)

        with tf.Session() as sess:

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(log_path + '/train',
                                                 sess.graph)
            # s_test_writer = tf.summary.FileWriter(log_path + '/source_test')
            # t_test_writer = tf.summary.FileWriter(log_path + '/target_test')
            sess.run(tf.global_variables_initializer())

            gen_source_batch = batch_generator(
                source_train, batch_size / 2)
            gen_target_batch = batch_generator(
                target_train, batch_size / 2)
            gen_source_only_batch = batch_generator(
                source_train, batch_size)
            gen_target_only_batch = batch_generator(
                target_train, batch_size)
            domain_labels = np.vstack([np.tile([1., 0.], [batch_size / 2, 1]),
                                       np.tile([0., 1.], [batch_size / 2, 1])])
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            for i in range(num_steps):

                # Adaptation param and learning rate schedule as described in the paper
                p = float(i) / num_steps
                l = 2. / (1. + np.exp(-10. * p)) - 1
                # lr = 0.01 / (1. + 10 * p) ** 0.75

                # Training step
                if training_mode == 'dann':

                    X0, y0 = gen_source_batch.next()
                    X1, y1 = gen_target_batch.next()

                    X_batch = np.vstack([X0, X1])
                    y_batch = np.vstack([y0, y1])

                    _, batch_loss, dloss, lloss, d_acc, l_acc, summary = \
                        sess.run([dann_train_op, total_loss, domain_loss, label_loss, domain_acc, label_acc, merged],
                                 feed_dict={X: X_batch, Y: y_batch, D_Y: domain_labels,
                                            is_train: True, sigma: l, prok: 0.7, is_training: True})
                    train_writer.add_summary(summary, i)

                    checkpoint = 1000

                    if (i+1)%checkpoint ==0 or i==num_steps-1:
                        checkpoint_path = os.path.join(train_saved_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=i)

                    if i % summary_step == 0 or i == num_steps - 1:
                        print("...Step {}".format(i))
                        print ' ...dann train loss: %f domain_loss %f, label_loss %f, domain_acc: %f  label_acc: %f' % \
                              (batch_loss, dloss, lloss, d_acc, l_acc)
                        # source_acc= sess.run([label_acc],
                        #                       feed_dict={X: source_test[0], Y: source_test[1],
                        #                                  is_train: False, prok: 1.0, is_training: False})
                        # # s_test_writer.add_summary(s_summary, i)
                        #
                        # target_acc = sess.run([label_acc],
                        #                                  feed_dict={X: target_test[0], Y: target_test[1],
                        #                                             is_train: False, prok: 1.0, is_training: False})
                        # # t_test_writer.add_summary(t_summary, i)
                        # print ' ...Source test (Young) accuracy:', source_acc
                        # print ' ...Target test (Old) accuracy:', target_acc

                elif training_mode == 'source':
                    X_batch, y_batch = gen_source_only_batch.next()
                    _, batch_loss, l_acc, s_summs = sess.run([label_train_op, label_loss, label_acc, merged],
                                                               feed_dict={X: X_batch, Y: y_batch, is_train: False,
                                                                          sigma: l, prok: 0.7, is_training: True})
                    train_writer.add_summary(s_summs, i)



                    checkpoint = 1000

                    if (i+1)%checkpoint ==0 or i==num_steps-1:
                        checkpoint_path = os.path.join(train_saved_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=i)

                    if i % summary_step == 0 or i == num_steps - 1:
                        print("...Step {}".format(i))
                        print(" ... source train loss: {}, accuracy: {}".format(batch_loss, l_acc))
                        # source_acc = sess.run([label_acc],
                        #                       feed_dict={X: source_test[0], Y: source_test[1],
                        #                                  is_train: False, prok: 1.0, is_training: False})
                        # # s_test_writer.add_summary(s_sums, i)
                        #
                        # target_acc = sess.run([label_acc],
                        #                                  feed_dict={X: target_test[0], Y: target_test[1],
                        #                                             is_train: False, prok: 1.0, is_training: False})
                        # # t_test_writer.add_summary(t_sums, i)
                        # print ' ...Source test (Young) accuracy:', source_acc
                        # print ' ...Target test (Old) accuracy:', target_acc
                elif training_mode == 'target':
                    X_batch, y_batch = gen_target_only_batch.next()
                    _, batch_loss = sess.run([label_train_op, label_loss],
                                             feed_dict={X: X_batch, Y: y_batch, is_train: False,
                                                        sigma: l, prok: 0.7})

                else:
                    raise ValueError("No such mode")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="domain adaptation")
    parser.add_argument('--mode', type=str, help='source, target, dann')
    parser.add_argument('--version', type=str, help='model version')
    parser.add_argument('--gpu', type=int, default=0, help='assign task to selected gpu')
    args = parser.parse_args()

    datasets_path = 'datasets'
    UniMiB_path = os.path.join(datasets_path, 'normalized_UbiMiB.cp')
    Opportunity_path = os.path.join(datasets_path, 'Opportunity', 'S1S2.cpkl')


    Realistics_cross_displacement_path = os.path.join(datasets_path, 'Realistics', 'cross_displacement.cpkl')
    Realistics_cross_subject_path = os.path.join(datasets_path, 'Realistics', 'cross_subject_target-16.cpkl')

    train(args.mode, args.version, args.gpu, data_path=Realistics_cross_displacement_path)

