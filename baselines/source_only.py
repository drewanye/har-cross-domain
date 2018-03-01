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

# # model params for RealWorld cross subject
# channels = 63
# window_length = 300
# num_classes = 8
# num_domain_labels = 2
# final_length = 50
# cut_channels = channels
# kernel_sizes = [[2, 1], [3, 1]]
# strides = [[2, 1], [3, 1]]

# model params for RealWorld cross placement
# channels = 9
# window_length = 300
# num_classes = 8
# num_domain_labels = 2
# final_length = 50
# cut_channels = channels
# kernel_sizes = [[2, 1], [3, 1]]
# strides = [[2, 1], [3, 1]]

# train params
batch_size = 32
num_filter = 64
num_units_lstm = 128
num_steps = 20000
summary_step = 200
checkpoint_step = 200
start_learning_rate = 0.0003
lr_decay_steps = 500
lr_decay_rate = 0.95

def create_placeholder():
    X = tf.placeholder(shape=[None, window_length, channels], dtype=tf.float32)
    Y = tf.placeholder(shape=[None, num_classes], dtype=tf.float32)
    prok = tf.placeholder(dtype=tf.float32)
    is_training = tf.placeholder(dtype=tf.bool)
    return X, Y, prok, is_training

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

def CNN_block(net, num_filters, residual=False):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
        net_in = net
        net = slim.conv2d(
            net,
            num_filters,
            stride=1,
            normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu)
        net = slim.conv2d(
            net,
            num_filters,
            stride=1,
            normalizer_fn=slim.batch_norm,
            activation_fn=None)
        net_out = net
        if residual:
            if net_in.get_shape()[-1] != num_filters:
                net_in = slim.conv2d(
                    net_in,
                    num_filters,
                    stride=1,
                    normalizer_fn=slim.batch_norm,
                    activation_fn=None)
            net_out = tf.nn.relu(net + net_in)

        return net_out

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

def batch_norm_params(is_training, batch_norm_decay):
    return {'is_training': is_training,
            'decay': batch_norm_decay,
            'epsilon': 0.001}

def get_losses(label_logits, labels):
    # label loss
    label_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=label_logits, labels=labels))

    tf.summary.scalar("label_loss", label_loss)

    return label_loss

def get_accuracies(label_logits, labels):
    label_prediction = tf.equal(tf.argmax(label_logits, 1), tf.argmax(labels, 1))
    label_acc = tf.reduce_mean(tf.cast(label_prediction, tf.float32))
    tf.summary.scalar("label_accuracy", label_acc)

    return label_acc, label_prediction

def create_model(x, is_training, prok):
    with tf.variable_scope('source_only'):
        X = tf.expand_dims(x, axis=3)  # batch_size, seq_length, channels, 1
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            # activation_fn=tf.nn.relu,
                            normalizer_params=batch_norm_params(is_training, 0.9),
                            weights_initializer=tf.random_normal_initializer(stddev=0.02),
                            weights_regularizer=tf.contrib.layers.l2_regularizer(1e-5)):
            with slim.arg_scope([slim.conv2d], kernel_size=[3, 3]):
                with slim.arg_scope([slim.max_pool2d], padding='SAME'):
                    net1 = CNN_block(X, 32)
                    net2 = CNN_block(net1, 32)

                    pool_net1 = slim.max_pool2d(net2, kernel_sizes[0], stride=strides[0], scope='pool1')

                    net3 = CNN_block(pool_net1, 64)
                    net4 = CNN_block(net3, 64)

                    all_features = slim.max_pool2d(net4, kernel_sizes[1], stride=strides[1], scope='pool2')



            label_logits = label_predictor(all_features, prok)

        return label_logits

def slip_eval(sess, X, Y, prok, is_training, predictions, test_data, slip_size):
    test_size = test_data[0].shape[0]
    start = 0
    end = 0
    total_preds = np.empty((0))
    while True:
        if test_size - start < slip_size:
            # print("... start {} -> end {}".format(start, test_size))
            slip_data = [i[start:] for i in test_data]
            pred = sess.run(predictions, feed_dict={
                X: slip_data[0], Y: slip_data[1], prok: 1.0, is_training: False})
            total_preds = np.concatenate((total_preds, pred))
            break
        else:
            end = start+slip_size
            # print("... start {} -> end {}".format(start, end))
            slip_data = [j[start:end]for j in test_data]
            start = end

            pred = sess.run(predictions, feed_dict={
                    X: slip_data[0], Y: slip_data[1], prok: 1.0, is_training: False})
            total_preds = np.concatenate((total_preds, pred))



    # if total_preds.shape[0] != test_size:
    #     raise ValueError("result is not equal to the initial")
    accuracy = np.mean(total_preds)

    return accuracy

def train(version, gpu, data_path):
    with tf.Graph().as_default():
        # create place_holder
        X, Y, prok, is_training = create_placeholder()
        # create model
        label_logits = create_model(X, is_training, prok)

        # define loss
        label_loss = get_losses(label_logits, Y)

        # define accuracy
        label_acc, label_predictions = get_accuracies(label_logits, Y)

        # train operations

        global_step = slim.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(
            start_learning_rate,
            global_step,
            decay_steps=lr_decay_steps,
            decay_rate=lr_decay_rate,
            staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        label_var_list, label_update_ops = data_utils.get_vars_and_update_ops('source_only')
        label_train_op = slim.learning.create_train_op(
            label_loss,
            optimizer,
            update_ops=label_update_ops,
            variables_to_train=label_var_list,
            clip_gradient_norm=5,
            summarize_gradients=False)

        source_train, source_test, target_train, target_test = data_utils.load_data(data_path)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        tf_config.gpu_options.allow_growth = True
        log_path = os.path.join('logs', version)

        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        # saver = tf.train.Saver()
        saver = tf.train.Saver(slim.get_variables(), max_to_keep=50)
        train_saved_dir = os.path.join('saved_checkpoint', version)

        with tf.Session(config=tf_config) as sess:

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(log_path + '/train',
                                                 sess.graph)
            source_test_writer = tf.summary.FileWriter(log_path + '/source_test')
            target_test_writer = tf.summary.FileWriter(log_path + '/target_test')
            sess.run(tf.global_variables_initializer())

            gen_source_batch = data_utils.batch_generator(
                source_train, batch_size)
            gen_target_batch = data_utils.batch_generator(
                target_train, batch_size)

            for i in range(num_steps):

                # Adaptation param and learning rate schedule as described in the paper
                p = float(i) / num_steps
                l = 2. / (1. + np.exp(-10. * p)) - 1
                # lr = 0.01 / (1. + 10 * p) ** 0.75

                # Training step

                X0, y0, _ = gen_source_batch.next()

                _, lloss, l_acc, summary = \
                    sess.run([label_train_op, label_loss, label_acc, merged],
                             feed_dict={X: X0, Y: y0, prok: 0.7, is_training: True})
                train_writer.add_summary(summary, i)



                if i%checkpoint_step ==0 or i==num_steps-1:
                    # checkpoint_path = os.path.join(train_saved_dir, 'model.ckpt')
                    # saver.save(sess, checkpoint_path, global_step=i)
                    print("...Step {}".format(i))
                    source_test_acc = slip_eval(sess, X, Y, prok, is_training, label_predictions, source_test, slip_size=100)
                    s_summary = tf.Summary()
                    s_summary.value.add(tag='label_accuracy', simple_value=source_test_acc)
                    source_test_writer.add_summary(s_summary, i)

                    target_test_acc = slip_eval(sess, X, Y, prok, is_training, label_predictions, target_test, slip_size=100)
                    t_summary = tf.Summary()
                    t_summary.value.add(tag='label_accuracy', simple_value=target_test_acc)
                    target_test_writer.add_summary(t_summary, i)

                    print(" ...source only test: source test {}, target test {}".format(source_test_acc,
                                                                                        target_test_acc))

                if i % summary_step == 0 or i == num_steps - 1:
                    print("...Step {}".format(i))
                    print ' ...source only train: label_loss %f, label_acc: %f' % \
                          (lloss, l_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="domain adaptation")
    parser.add_argument('--version', type=str, help='model version')
    parser.add_argument('--gpu', type=int, default=0, help='assign task to selected gpu')
    args = parser.parse_args()

    datasets_path = 'datasets'
    UniMiB_path = os.path.join(datasets_path, 'normalized_UbiMiB.cp')
    Opportunity_path = os.path.join(datasets_path, 'Opportunity', 'S1S2.cpkl')


    Realistics_cross_displacement_path = os.path.join(datasets_path, 'Realistics', 'cross_displacement.cpkl')
    Realistics_cross_subject_path = os.path.join(datasets_path, 'Realistics', 'cross_subject_target-16.cpkl')

    Realworld_cross_subject_9_path = os.path.join(datasets_path, 'RealWorld', 'cross_subject_9.cpkl')
    Realworld_cross_subject_5_path = os.path.join(datasets_path, 'RealWorld', 'cross_subject_5.cpkl')


    Realworld_uparm_forarm_path = os.path.join(datasets_path, 'RealWorld', 'cross_placement_upperarm_forearm.cpkl')
    Realworld_thigh_shin_path = os.path.join(datasets_path, 'RealWorld', 'cross_placement_thigh_shin.cpkl')
    Realworld_chest_waist_path = os.path.join(datasets_path, 'RealWorld', 'cross_placement_chest_waist.cpkl')
    Realworld_ideal_to_dislocation_path = data_utils.get_data('RealWorld', 'ideal_to_dislocation')

    Realistics_target_users_01_path = os.path.join(datasets_path, 'Realistics', 'user_data-01.cpkl')

    train(args.version, args.gpu, data_path=Realistics_target_users_01_path)