__author__ = 'Zhenan Ye'

import tensorflow as tf
import numpy as np
import cPickle as cpkl
import os
import zipfile
from io import BytesIO
import utils as skutils

slim = tf.contrib.slim


def get_data(dataset, type):
    datasets_path = 'datasets'
    if dataset == 'Realistics':
        if type == 'cross_placement':
            return os.path.join(datasets_path, dataset, 'cross_displacement.cpkl')
        elif type == 'cross_subject':
            return os.path.join(datasets_path, dataset, 'cross_subject_target-16.cpkl')
        else:
            raise ValueError("dataset {} have no data file {}".format(dataset, type))
    elif dataset == 'RealWorld':
        if type == 'cross_subject_5':
            return os.path.join(datasets_path, dataset, 'cross_subject_5.cpkl')
        elif type == 'cross_subject_9':
            return os.path.join(datasets_path, dataset, 'cross_subject_9.cpkl')
        elif type == 'upperarm_to_forearm':
            return os.path.join(datasets_path, dataset, 'cross_placement_upperarm_forearm.cpkl')
        elif type == 'forearm_to_shin':
            return os.path.join(datasets_path, dataset, 'cross_placement_forearm_shin.cpkl')
        elif type == 'ideal_to_dislocation':
            return os.path.join(datasets_path, dataset, 'cross_placement_ideal_dislocation.cpkl')
        else:
            raise ValueError("dataset {} have no data file {}".format(dataset, type))
    else:
        raise ValueError("There is no dataset {}".format(dataset))



def show_label_distribution(func):
    def wrapper(*args, **kwargs):
        source_train, source_test, target_train, target_test = func(*args, **kwargs)
        def print_label_distribution(train_data):
            y = np.where(train_data[1] == 1)[1]
            items, counts = np.unique(y, return_counts=True)
            print("... iterms:{}, counts: {}".format(items, counts))
        print(" ... source_train:")
        print_label_distribution(source_train)
        print(" ... source_test:")
        print_label_distribution(source_test)
        print(" ... target_train:")
        print_label_distribution(target_train)
        print(" ... target_test:")
        print_label_distribution(target_test)
        return source_train, source_test, target_train, target_test
    return wrapper

@show_label_distribution
def load_data(data_path):
    print("...Loading data from {}".format(data_path))
    with open(data_path, 'r') as f:
        data = cpkl.load(f)
    source_data, target_data = data
    print("...the size of source data: {}, the size of target data: {}".format(
        source_data[0].shape, target_data[0].shape))

    # split source data
    source_size = source_data[0].shape[0]
    source_train_size = int(source_size * 0.7)
    source_train = [i[:source_train_size] for i in source_data]
    source_test = [i[source_train_size:] for i in source_data]
    print(" ...source train size: {}, source test size: {}".format(
        source_train[0].shape, source_test[0].shape))

    # split target data
    target_size = target_data[0].shape[0]
    target_train_size = int(target_size * 0.7)
    target_train = [i[:target_train_size] for i in target_data]
    target_test = [i[target_train_size:] for i in target_data]
    print(" ...target train size: {}, target test size: {}".format(
        target_train[0].shape, target_test[0].shape))

    return source_train, source_test, target_train, target_test

def normalize_data(source_data, target_data):
    '''
    :param source_data: [nums, seq_length, channels]
    :param target_data: [nums, seq_length, channels]
    :return: normalized young and old data
    '''
    shape = source_data.shape
    ax = 0
    if len(shape) == 3:
        ax = (0, 1)
    all_data = np.vstack([source_data, target_data])
    mean = np.mean(all_data, axis=ax)
    std = np.std(all_data, axis=ax)
    norm_source_data = (source_data-mean)/std
    norm_target_data = (target_data-mean)/std
    return norm_source_data, norm_target_data

def one_hot(y_, n_values):
    # y_ = y_.reshape(len(y_))
    # n_values = int(np.max(y_))+1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def convert_seq_data(data_x, data_y, seq_length):
    '''
    [n, channels] -> [m, seq_length, channels]
    :param data_path:
    :return:
    '''

    data_size = data_x.shape[0]
    channels = data_x.shape[1]

    # round data with seq_length
    if data_size < seq_length:
        raise ValueError("seq_length is less than data size")
    round_data_size = data_size / seq_length * seq_length

    data_x = data_x[:round_data_size]
    data_y = data_y[:round_data_size]

    def slide_window(x, y):
        '''
        slide window without overlapping
        :param x:
        :param y:
        :return:
        '''
        num_x = x.shape[0] / seq_length * seq_length
        seq_x = np.reshape(x[:num_x], newshape=(-1, seq_length, channels))
        seq_y = np.reshape(y[:num_x], newshape=(-1, seq_length))

        return seq_x, seq_y

    def define_label(units, option='MAX'):
        if option == 'LAST' or option is None:
            return units[-1]
        elif option == 'MAX':
            items, counts = np.unique(units, return_counts=True)
            return items[np.argmax(counts)]
        else:
            raise ValueError("option parameter is invalid")

    seq_x, seq_y = slide_window(data_x, data_y)
    seq_y = [define_label(i) for i in seq_y]
    seq_y = np.asarray(seq_y)

    return seq_x, seq_y

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

def get_vars_and_update_ops(scope):
    # is_trainable = lambda x: x in tf.trainable_variables()
    # var_list = filter(is_trainable, slim.get_model_variables(scope))

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

    tf.logging.info('All variables for scope: %s',
                  scope)
    tf.logging.info('Trainable variables for scope: %s', var_list)

    return var_list, update_ops

def lrelu(x, leakiness=0.2):
  """Relu, with optional leaky support."""
  return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
