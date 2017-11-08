__author__ = 'Zhenan Ye'

import tensorflow as tf
import numpy as np
import cPickle as cp
import os
import utils
from flip_gradient import flip_gradient
import tqdm
from sklearn import utils as skutils
import argparse
from tensorflow.python import debug as tf_debug

datasets_path = 'datasets'
all_UniMiB_path = os.path.join(datasets_path, 'all_UbiMiB.cp')


def load_data():
    print("...Loading data")
    with open(all_UniMiB_path, 'rb') as f:
        data = cp.load(f)
    source_data, target_data = data
    print("...the size of source data: {}, the size of target data: {}".format(
        source_data[0].shape, target_data[0].shape))
    # split data into train and test
    source_x_data, source_y_data = source_data
    target_x_data, target_y_data = target_data
    source_x_data, source_y_data = skutils.shuffle(source_x_data, source_y_data, random_state=0)
    target_x_data, target_y_data = skutils.shuffle(target_x_data, target_y_data, random_state=0)

    # split source data
    source_size = source_x_data.shape[0]
    source_train_size = int(source_size * 0.7)
    source_x_train = source_x_data[:source_train_size]
    source_y_train = source_y_data[:source_train_size]
    source_x_test = source_x_data[source_train_size:]
    source_y_test = source_y_data[source_train_size:]
    print(" ...source train size: {}, source test size: {}".format(
        source_x_train.shape, source_x_test.shape))

    # split target data
    target_size = target_x_data.shape[0]
    target_train_size = int(target_size * 0.7)
    target_x_train = target_x_data[:target_train_size]
    target_y_train = target_y_data[:target_train_size]
    target_x_test = target_x_data[target_train_size:]
    target_y_test = target_y_data[target_train_size:]
    print(" ...target train size: {}, target test size: {}".format(
        target_x_train.shape, target_x_test.shape))

    return (source_x_train, source_y_train), (source_x_test, source_y_test),\
           (target_x_train, target_y_train), (target_x_test, target_y_test)

# params
channels = 3
batch_size = 100
window_length = 151
num_labels = 17
num_domain_labels = 2
filter_size = 5
num_filter = 64
# feature_extractor 27, residual_feat_extractor 16,26
final_length = 26
num_units_lstm = 128
num_steps = 10000

X = tf.placeholder(shape=[None, window_length, channels], dtype=tf.float32)
Y = tf.placeholder(shape=[None, num_labels], dtype=tf.float32)
D_Y = tf.placeholder(shape=[None, num_domain_labels], dtype=tf.float32)
prok = tf.placeholder(dtype=tf.float32)
learning_rate = tf.placeholder(dtype=tf.float32)
is_train = tf.placeholder(dtype=tf.bool)
sigma = tf.placeholder(dtype=tf.float32)


def residual(x, in_channel, out_channel):
    """residual unit with 2 layers
    convolution:
        width filter: 1
        height filter: 3
    """
    orig_x = x
    width_filter = 3
    height_filter = 1
    with tf.variable_scope('conv1'):
        conv1 = utils.conv(x, [width_filter, height_filter, in_channel, out_channel], [out_channel], padding='SAME')
        relu1 = utils.activation(conv1)
    with tf.variable_scope('conv2'):
        conv2 = utils.conv(relu1, [width_filter, height_filter, out_channel, out_channel], [out_channel], padding='SAME')
    with tf.variable_scope('add'):
        if in_channel != out_channel:
            orig_x = utils.conv(x, [width_filter, height_filter, in_channel, out_channel], [out_channel], padding='SAME')

    return utils.activation(conv2 + orig_x)

def feature_extractor(x):
    x = tf.reshape(x, [-1, window_length, channels, 1])
    with tf.variable_scope('conv1'):
        conv1 = utils.conv(x, [filter_size, 1, 1, num_filter], [num_filter],
                           padding='VALID')
        relu1 = utils.activation(conv1)
    with tf.variable_scope('conv2'):
        conv2 = utils.conv(relu1, [filter_size, 1, num_filter, num_filter], [num_filter],
                           padding='VALID')
        relu2 = utils.activation(conv2)
    with tf.variable_scope('conv3'):
        conv3 = utils.conv(relu2, [filter_size, 1, num_filter, num_filter], [num_filter],
                           padding='VALID')
        relu3 = utils.activation(conv3)
    with tf.variable_scope('conv4'):
        conv4 = utils.conv(relu3, [filter_size, 1, num_filter, num_filter], [num_filter],
                           padding='VALID')
        relu4 = utils.activation(conv4)
    with tf.variable_scope('pool1'):
        h_pool1 = utils.max_pool(relu4, 5, 1, 5, 1, padding='SAME')

    return h_pool1

def residual_feat_extractor(x):
    x = tf.reshape(x, [-1, window_length, channels, 1])
    x_shape = x.get_shape()
    with tf.variable_scope('residual1'):
        r1 = residual(x, x_shape[-1], 32)
        # tf.summary.histogram('res_output1', r1)
    with tf.variable_scope('residual2'):
        r2 = residual(r1, r1.get_shape()[-1], 32)
        # tf.summary.histogram('res_output2', r2)

    with tf.variable_scope('pool0'):
        h_pool0 = utils.max_pool(r2, 2, 1, 2, 1, padding='SAME')

    with tf.variable_scope('residual3'):
        r3 = residual(h_pool0, h_pool0.get_shape()[-1], 64)
        # tf.summary.histogram('res_output3', r3)
    with tf.variable_scope('residual4'):
        r4 = residual(r3, r3.get_shape()[-1], 64)
        # tf.summary.histogram('res_output4', r4)

    with tf.variable_scope('pool1'):
        h_pool1 = utils.max_pool(r4, 3, 1, 3, 1, padding='SAME')
    return h_pool1

def label_predictor(features):
    features = tf.transpose(features, [1, 0, 2, 3])  # (135, batch_size, 3, 64)

    feat_flat = tf.reshape(features, [-1, channels * num_filter])

    lstm_inputs = tf.split(feat_flat, num_or_size_splits=final_length, axis=0)
    with tf.variable_scope("lstm_layers"):
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(num_units_lstm, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_1 = tf.contrib.rnn.DropoutWrapper(lstm_cell_1, input_keep_prob=prok, output_keep_prob=prok)
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units_lstm, forget_bias=1.0, state_is_tuple=True)
        # lstm_cell_2 = tf.contrib.rnn.DropoutWrapper(lstm_cell_2, input_keep_prob=self.prok, output_keep_prob=self.prok)
        cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        outputs, _ = tf.contrib.rnn.static_rnn(cells, lstm_inputs, dtype=tf.float32)
    pred_y = utils.scores(outputs[-1], [num_units_lstm, num_labels], [num_labels])
    return pred_y

def _add_layer(x):
    x = tf.reshape(x, [-1, window_length, channels, 1])
    with tf.variable_scope('conv1'):
        conv1 = utils.conv(x, [filter_size, 1, 1, num_filter], [num_filter],
                           padding='VALID')
        relu1 = utils.activation(conv1)
    with tf.variable_scope('conv2'):
        conv2 = utils.conv(relu1, [filter_size, 1, num_filter, num_filter], [num_filter],
                           padding='VALID')
        relu2 = utils.activation(conv2)
    with tf.variable_scope('conv3'):
        conv3 = utils.conv(relu2, [filter_size, 1, num_filter, num_filter], [num_filter],
                           padding='VALID')
        relu3 = utils.activation(conv3)
    with tf.variable_scope('conv4'):
        conv4 = utils.conv(relu3, [filter_size, 1, num_filter, num_filter], [num_filter],
                           padding='VALID')
        relu4 = utils.activation(conv4)

    relu4 = tf.transpose(relu4, [1, 0, 2, 3])  # (135, batch_size, 3, 64)

    relu4_flat = tf.reshape(relu4, [-1, channels * num_filter])

    lstm_inputs = tf.split(relu4_flat, num_or_size_splits=final_length, axis=0)
    with tf.variable_scope("lstm_layers"):
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(num_units_lstm, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_1 = tf.contrib.rnn.DropoutWrapper(lstm_cell_1, input_keep_prob=prok, output_keep_prob=prok)
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units_lstm, forget_bias=1.0, state_is_tuple=True)
        # lstm_cell_2 = tf.contrib.rnn.DropoutWrapper(lstm_cell_2, input_keep_prob=self.prok, output_keep_prob=self.prok)
        cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        outputs, _ = tf.contrib.rnn.static_rnn(cells, lstm_inputs, dtype=tf.float32)
    return outputs

def lstm_only(x):
    transposed_x = tf.transpose(x, [1, 0, 2])
    x_flat = tf.reshape(transposed_x, [-1, channels])
    lstm_inputs = tf.split(x_flat, num_or_size_splits=window_length, axis=0)
    with tf.variable_scope("lstm_layers"):
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(num_units_lstm, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_1 = tf.contrib.rnn.DropoutWrapper(lstm_cell_1, input_keep_prob=prok, output_keep_prob=prok)
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units_lstm, forget_bias=1.0, state_is_tuple=True)
        # lstm_cell_2 = tf.contrib.rnn.DropoutWrapper(lstm_cell_2, input_keep_prob=self.prok, output_keep_prob=self.prok)
        cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        outputs, _ = tf.contrib.rnn.static_rnn(cells, lstm_inputs, dtype=tf.float32)
    return outputs


def build_model():
    print("...Building model")
    x = utils.input_batch_norm(X)

    with tf.variable_scope('feature_extractor'):
        # outputs = _add_layer(x)
        # outputs = lstm_only(x)
        # all_features = feature_extractor(x)
        all_features = residual_feat_extractor(x)
        # all_features = outputs[-1]
    with tf.variable_scope('label_predictor'):
        source_features = tf.slice(all_features, [0, 0, 0, 0], [batch_size/2, -1, -1, -1])
        classfier_features = tf.cond(is_train, lambda: source_features, lambda: all_features)
        all_labels = Y
        source_labels = tf.slice(all_labels, [0, 0], [batch_size/2, -1])
        labels = tf.cond(is_train, lambda: source_labels, lambda: all_labels)
        pred_y = label_predictor(classfier_features)
        # with tf.variable_scope("full1"):
        #     f1 = tf.nn.relu(utils.full_conn(classfier_features, [num_units_lstm, 64], [64]))
        # with tf.variable_scope('full2'):
        #     f2 = tf.nn.relu(utils.full_conn(f1, [64, 64], [64]))
        # with tf.variable_scope('score'):
        #     pred_y = utils.scores(f2, [64, 17], [17])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_y, labels=labels))
        loss_summary = tf.summary.scalar("predict_loss", loss)
        correct_prediction = tf.equal(tf.argmax(pred_y, 1), tf.argmax(labels, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_summary = tf.summary.scalar("predict_accuracy", acc)
    with tf.variable_scope('domain_predictor'):
        all_features_flat = tf.reshape(all_features, [-1, final_length*channels*num_filter])
        feat = flip_gradient(all_features_flat, sigma)
        with tf.variable_scope('full1'):
            d_f1 = utils.full_conn(feat, [final_length*channels*num_filter, 1024], [1024])
        with tf.variable_scope('full2'):
            d_f2 = utils.full_conn(d_f1, [1024, 128], [128])
        with tf.variable_scope('score'):
            domain_pred_y = utils.scores(d_f2, [128, 2], [2])
        domain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=tf.clip_by_value(domain_pred_y,1e-10,1.0), labels=D_Y))
        tf.summary.scalar('domain_loss', domain_loss)
        domain_correct_pred = tf.equal(tf.argmax(domain_pred_y, 1), tf.argmax(D_Y, 1))
        domain_acc = tf.reduce_mean(tf.cast(domain_correct_pred, tf.float32))
        tf.summary.scalar('domain_accuracy', domain_acc)
    total_loss = loss + domain_loss
    tf.summary.scalar('total_loss', total_loss)
    regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
    dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)
    tf.summary.scalar('learning_rate', learning_rate)


    return loss, domain_loss, total_loss, acc, domain_acc, \
           regular_train_op, dann_train_op, [loss_summary, accuracy_summary]

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


def train(loss, domain_loss, total_loss, acc, domain_acc, regular_train_op, dann_train_op, selected_summaries, training_mode, version, gpu):
    print("...Training")
    source_train, source_test, target_train, target_test = load_data()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = os.path.join('logs', version)
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        selected_merged = tf.summary.merge(selected_summaries)
        train_writer = tf.summary.FileWriter(log_path + '/train',
                                             sess.graph)
        test_writer = tf.summary.FileWriter(log_path + '/test')
        sess.run(tf.global_variables_initializer())
        gen_source_batch = batch_generator(
            list(source_train), batch_size / 2)
        gen_target_batch = batch_generator(
            list(target_train), batch_size / 2)
        gen_source_only_batch = batch_generator(
            list(source_train), batch_size)
        gen_target_only_batch = batch_generator(
            list(target_train), batch_size)
        domain_labels = np.vstack([np.tile([1., 0.], [batch_size / 2, 1]),
                                   np.tile([0., 1.], [batch_size / 2, 1])])
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        for i in tqdm.tqdm(range(num_steps)):

            # Adaptation param and learning rate schedule as described in the paper
            p = float(i) / num_steps
            l = 2. / (1. + np.exp(-10. * p)) - 1
            lr = 0.01 / (1. + 10 * p) ** 0.75

            # Training step
            if training_mode == 'dann':

                X0, y0 = gen_source_batch.next()
                X1, y1 = gen_target_batch.next()
                X_batch = np.vstack([X0, X1])
                y_batch = np.vstack([y0, y1])

                _, batch_loss, dloss, ploss, d_acc, p_acc, summary = \
                    sess.run([dann_train_op, total_loss, domain_loss, loss, domain_acc, acc, merged],
                             feed_dict={X: X_batch, Y: y_batch, D_Y: domain_labels,
                                        is_train: True, sigma: l, learning_rate: lr, prok: 0.7})
                train_writer.add_summary(summary, i)
                if i % 100 == 0 or i == num_steps -1:
                    print("...Step {}".format(i))
                    print ' ...loss: %f  d_acc: %f  p_acc: %f  p: %f  l: %f  lr: %f' % \
                          (batch_loss, d_acc, p_acc, p, l, lr)
                    source_acc = sess.run(acc,
                                          feed_dict={X: source_test[0], Y: source_test[1],
                                                     is_train: False, prok: 1.0})

                    target_acc, t_summary = sess.run([acc, selected_merged],
                                          feed_dict={X: target_test[0], Y: target_test[1],
                                                     is_train: False, prok: 1.0})
                    test_writer.add_summary(t_summary, i)
                    print ' ...Source (Young) accuracy:', source_acc
                    print ' ...Target (Old) accuracy:', target_acc

            elif training_mode == 'source':
                X_batch, y_batch = gen_source_only_batch.next()
                _, batch_loss, p_acc, s_summary = sess.run([regular_train_op, loss, acc, selected_merged],
                                         feed_dict={X: X_batch, Y: y_batch, is_train: False,
                                                    sigma: l, learning_rate: lr, prok: 0.7})
                train_writer.add_summary(s_summary, i)

                if i % 100 == 0 or i == num_steps -1:
                    print("...Step {}".format(i))
                    print(" .. loss: {}, accuracy: {}".format(batch_loss, p_acc))
                    source_acc = sess.run(acc, feed_dict={X: source_test[0], Y: source_test[1],is_train: False, prok: 1.0})

                    target_acc, t_summary = sess.run([acc, selected_merged],
                                                     feed_dict={X: target_test[0], Y: target_test[1],
                                                                is_train: False, prok: 1.0})
                    test_writer.add_summary(t_summary, i)
                    print ' ...Source (Young) accuracy:', source_acc
                    print ' ...Target (Old) accuracy:', target_acc
            elif training_mode == 'target':
                X_batch, y_batch = gen_target_only_batch.next()
                _, batch_loss = sess.run([regular_train_op, loss],
                                         feed_dict={X: X_batch, Y: y_batch, is_train: False,
                                                    sigma: l, learning_rate: lr, prok: 0.7})

        # # Compute final evaluation on test data
        # source_acc = sess.run(acc,
        #                       feed_dict={X: source_test[0], Y: source_test[1],
        #                                  is_train: False, prok: 1.0})
        #
        # target_acc = sess.run(acc,
        #                       feed_dict={X: target_test[0], Y: target_test[1],
        #                                  is_train: False, prok: 1.0})

        # test_domain_acc = sess.run(domain_acc,
        #                            feed_dict={model.X: combined_test_imgs,
        #                                       model.domain: combined_test_domain, model.l: 1.0})
        #
        # test_emb = sess.run(model.feature, feed_dict={model.X: combined_test_imgs})

    # return source_acc, target_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Activity Recognition using Deep Multi-task Learning")
    parser.add_argument('--mode', type=str, help='source, target, dann')
    parser.add_argument('--version', type=str, help='model version')
    parser.add_argument('--gpu', type=int, default=0, help='assign task to selected gpu')
    args = parser.parse_args()
    loss, domain_loss, total_loss, acc, domain_acc, regular_train_op, dann_train_op, summaries = build_model()
    # print '\nSource only training'
    # source_acc, target_acc = train(loss, domain_loss, total_loss, acc, domain_acc,
    #                                regular_train_op, dann_train_op, 'source', args.version, args.gpu)
    # print 'Source (Young) accuracy:', source_acc
    # print 'Target (Old) accuracy:', target_acc

    print '\nDomain adaptation training'
    train(loss, domain_loss, total_loss, acc, domain_acc,
                                   regular_train_op, dann_train_op, summaries, args.mode, args.version, args.gpu)
    # print 'Source (Young) accuracy:', source_acc
    # print 'Target (Old) accuracy:', target_acc
    # print 'Domain accuracy:', d_acc
