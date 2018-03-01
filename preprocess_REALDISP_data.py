__author__ = 'Zhenan Ye'

import numpy as np
import matplotlib.pyplot as plt
import os
import cPickle as cp
import utils
import zipfile
from io import BytesIO
from sklearn import utils as skutils
import data_utils

Num_Classes = 33
Seq_Length = 300
Channels = 117
Realistics_ideal_f = ['subject0{}_ideal.log'.format(i) for i in range(1, 18)]
Realistics_files = ['subject01_ideal.log',
                    'subject02_ideal.log',
                    'subject03_ideal.log',
                    'subject04_ideal.log',
                    'subject05_ideal.log',
                    'subject06_ideal.log',
                    'subject07_ideal.log',
                    'subject08_ideal.log',
                    'subject09_ideal.log',
                    'subject10_ideal.log',
                    'subject11_ideal.log',
                    'subject12_ideal.log',
                    'subject13_ideal.log',
                    'subject14_ideal.log',
                    'subject15_ideal.log',
                    'subject16_ideal.log',
                    'subject17_ideal.log',
                    'subject02_mutual4.log',
                    'subject02_mutual5.log',
                    'subject02_mutual6.log',
                    'subject02_mutual7.log',
                    'subject05_mutual4.log',
                    'subject05_mutual5.log',
                    'subject05_mutual6.log',
                    'subject05_mutual7.log',
                    'subject15_mutual5.log',
                    'subject15_mutual6.log',
                    'subject15_mutual7.log',
                    ''
                    ]


def process_realistics_data(data):
    ignored_colums = [0, 1]
    data = np.delete(data, ignored_colums, axis=1)
    x, y = divide_x_y(data)
    x, y = delete_zero_labels(x, y)

    return x, y

def divide_x_y(data):
    x = data[:, :117]
    y = data[:, 117]
    return x, y

def delete_zero_labels(x, y):
    s_zero_index = np.where(y == 0.0)
    x, y = (np.delete(i, s_zero_index, axis=0) for i in (x, y))
    print("delete zero labels: {}".format(len(s_zero_index[0])))
    return x, y


def get_cross_displacement_data(data_path, saved_data_path):
    zf = zipfile.ZipFile(data_path)

    # ideal as source data, mutual as target data
    source_x = np.empty((0, 117))
    source_y = np.empty((0))
    target_x = np.empty((0, 117))
    target_y = np.empty((0))

    for filename in Realistics_files:
        try:
            file_path = os.path.join('realistics', filename)
            data = np.loadtxt(BytesIO(zf.read(file_path)))
            data_x, data_y = process_realistics_data(data)
            if 'ideal' in filename:
                print('... file {} -> source data'.format(filename))
                source_x = np.vstack((source_x, data_x))
                source_y = np.concatenate((source_y, data_y))
            # elif 'self' in filename:
            #     print('... file {} -> source data'.format(filename))
            #     source_x = np.vstack((source_x, data_x))
            #     source_y = np.concatenate((source_y, data_y))
            elif 'mutual' in filename:
                print('... file{} -> target data'.format(filename))
                target_x = np.vstack((target_x, data_x))
                target_y = np.concatenate((target_y, data_y))
            else:
                print("... file {} not be contained".format(filename))

        except KeyError:
            print 'ERROR: Did not find {0} in zip file'.format(filename)

    # count lables
    s_items, s_counts = np.unique(source_y, return_counts=True)
    t_items, t_counts = np.unique(target_y, return_counts=True)
    print("... source iterms:{}, counts: {}".format(s_items, s_counts))
    print("... target iterms:{}, counts: {}".format(t_items, t_counts))
    # convert into sequence data
    source_x, source_y = data_utils.convert_seq_data(source_x, source_y, seq_length=Seq_Length)
    target_x, target_y = data_utils.convert_seq_data(target_x, target_y, seq_length=Seq_Length)

    # normalize
    source_x, target_x = data_utils.normalize_data(source_x, target_x)
    # shuffle
    source_x, source_y = skutils.shuffle(source_x, source_y, random_state=0)
    target_x, target_y = skutils.shuffle(target_x, target_y, random_state=0)
    # one hot encoding
    source_y = data_utils.one_hot(source_y - 1, Num_Classes)
    target_y = data_utils.one_hot(target_y - 1, Num_Classes)


    print("the size of source domain data: {}, {} | the size of target domain data: {}, {}".format(
        source_x.shape, source_y.shape, target_x.shape, target_y.shape))

    with open(saved_data_path, 'wb') as f:
        cp.dump(((source_x, source_y), (target_x, target_y)), f)

def get_cross_subject_data(data_path, saved_data_path, target_id):
    zf = zipfile.ZipFile(data_path)

    # ideal as source data, mutual as target data
    source_x = np.empty((0, Channels))
    source_y = np.empty((0))
    target_x = np.empty((0, Channels))
    target_y = np.empty((0))

    for filename in Realistics_files:
        try:
            file_path = os.path.join('realistics', filename)
            if 'ideal' in filename:
                data = np.loadtxt(BytesIO(zf.read(file_path)))
                data_x, data_y = process_realistics_data(data)

                subject_id = int(filename.split('_')[0][-2:])
                if subject_id == target_id:
                    print('... file {} -> target data'.format(filename))
                    target_x = np.vstack((target_x, data_x))
                    target_y = np.concatenate((target_y, data_y))
                else:
                    print('... file {} -> source data'.format(filename))
                    source_x = np.vstack((source_x, data_x))
                    source_y = np.concatenate((source_y, data_y))
        except KeyError:
            print 'ERROR: Did not find {0} in zip file'.format(filename)

    # convert into sequence data
    source_x, source_y = data_utils.convert_seq_data(source_x, source_y, seq_length=Seq_Length)
    target_x, target_y = data_utils.convert_seq_data(target_x, target_y, seq_length=Seq_Length)

    # normalize
    source_x, target_x = data_utils.normalize_data(source_x, target_x)
    # shuffle
    source_x, source_y = skutils.shuffle(source_x, source_y, random_state=0)
    target_x, target_y = skutils.shuffle(target_x, target_y, random_state=0)
    # one hot encoding
    source_y = data_utils.one_hot(source_y - 1, Num_Classes)
    target_y = data_utils.one_hot(target_y - 1, Num_Classes)

    print("the size of source domain data: {}, {} | the size of target domain data: {}, {}".format(
        source_x.shape, source_y.shape, target_x.shape, target_y.shape))

    with open(saved_data_path, 'wb') as f:
        cp.dump(((source_x, source_y), (target_x, target_y)), f)

def get_cross_mulit_subjects_data(data_path, saved_data_path, target_id):
    zf = zipfile.ZipFile(data_path)

    # ideal as source data, mutual as target data
    source_x_raw = np.empty((0, Channels))
    source_y = np.empty((0))
    s_user_labels = np.empty((0))
    target_x_raw = np.empty((0, Channels))
    target_y = np.empty((0))
    t_user_labels = np.empty((0))

    for filename in Realistics_files:
        try:
            file_path = os.path.join('realistics', filename)
            if 'ideal' in filename:
                data = np.loadtxt(BytesIO(zf.read(file_path)))
                data_x, data_y = process_realistics_data(data)

                subject_id = int(filename.split('_')[0][-2:])
                if subject_id == target_id:
                    print('... file {} -> target data'.format(filename))
                    target_x_raw = np.vstack((target_x_raw, data_x))
                    target_y = np.concatenate((target_y, data_y))
                    t_user_labels = np.concatenate((np.tile([subject_id], data_y.shape[0]), t_user_labels))
                else:
                    print('... file {} -> source data'.format(filename))
                    source_x_raw = np.vstack((source_x_raw, data_x))
                    source_y = np.concatenate((source_y, data_y))
                    s_user_labels = np.concatenate((np.tile([subject_id], data_y.shape[0]), s_user_labels))
        except KeyError:
            print 'ERROR: Did not find {0} in zip file'.format(filename)

    # convert into sequence data
    source_x, source_y = data_utils.convert_seq_data(source_x_raw, source_y, seq_length=Seq_Length)
    _, s_user_labels = data_utils.convert_seq_data(source_x_raw, s_user_labels, seq_length=Seq_Length)
    target_x, target_y = data_utils.convert_seq_data(target_x_raw, target_y, seq_length=Seq_Length)
    _, t_user_labels = data_utils.convert_seq_data(target_x_raw, t_user_labels, seq_length=Seq_Length)
    import pdb;pdb.set_trace()

    # normalize
    source_x, target_x = data_utils.normalize_data(source_x, target_x)
    # shuffle
    source_x, source_y, s_user_labels = skutils.shuffle(source_x, source_y, s_user_labels, random_state=0)
    target_x, target_y, t_user_labels = skutils.shuffle(target_x, target_y, t_user_labels,  random_state=0)
    # one hot encoding
    source_y = data_utils.one_hot(source_y - 1, Num_Classes)
    s_user_labels = data_utils.one_hot(s_user_labels-1, 17)
    target_y = data_utils.one_hot(target_y - 1, Num_Classes)
    t_user_labels = data_utils.one_hot(t_user_labels-1, 17)

    print("the size of source domain data: {}, {} | the size of target domain data: {}, {}".format(
        source_x.shape, source_y.shape, s_user_labels.shape, target_x.shape, target_y.shape, t_user_labels.shape))

    with open(saved_data_path, 'wb') as f:
        cp.dump(((source_x, source_y, s_user_labels), (target_x, target_y, t_user_labels)), f)

def get_user_data(data_path, saved_data_path, target_id):
    zf = zipfile.ZipFile(data_path)

    train_x_raw = np.empty((0, Channels))
    train_y = np.empty((0))
    user_labels = np.empty((0))
    test_x = np.empty((0, Channels))
    test_y = np.empty((0))
    user_id = 1
    for filename in Realistics_files:
        try:
            file_path = os.path.join('realistics', filename)
            if 'ideal' in filename:
                data = np.loadtxt(BytesIO(zf.read(file_path)))
                data_x, data_y = process_realistics_data(data)

                subject_id = int(filename.split('_')[0][-2:])
                if subject_id == target_id:
                    print('... file {} -> target data'.format(filename))
                    test_x = np.vstack((test_x, data_x))
                    test_y = np.concatenate((test_y, data_y))
                else:
                    print('... file {} -> source data'.format(filename))
                    train_x_raw = np.vstack((train_x_raw, data_x))
                    train_y = np.concatenate((train_y, data_y))
                    user_labels = np.concatenate((np.tile([user_id], data_y.shape[0]), user_labels))
                    user_id = user_id + 1
        except KeyError:
            print 'ERROR: Did not find {0} in zip file'.format(filename)

            # convert into sequence data
    train_x, train_y = data_utils.convert_seq_data(train_x_raw, train_y, seq_length=Seq_Length)
    tx, user_labels = data_utils.convert_seq_data(train_x_raw, user_labels, seq_length=Seq_Length)
    test_x, test_y = data_utils.convert_seq_data(test_x, test_y, seq_length=Seq_Length)

    # normalize
    train_x, test_x = data_utils.normalize_data(train_x, test_x)
    # shuffle
    train_x, train_y, user_labels = skutils.shuffle(train_x, train_y, user_labels, random_state=0)
    test_x, test_y = skutils.shuffle(test_x, test_y, random_state=0)
    # one hot encoding
    train_y = data_utils.one_hot(train_y - 1, Num_Classes)
    test_y = data_utils.one_hot(test_y - 1, Num_Classes)
    user_labels = data_utils.one_hot(user_labels - 1, 16)


    print("the size of source domain data: {}, {} | the size of target domain data: {}, {}".format(
        train_x.shape, train_y.shape, user_labels.shape, test_x.shape, test_y.shape))

    with open(saved_data_path, 'wb') as f:
        cp.dump(((train_x, train_y, user_labels), (test_x, test_y)), f)



if __name__=='__main__':
    dataset_path = 'datasets'
    realistics_path = os.path.join(dataset_path, 'realistics.zip')
    # saved_data_path = "datasets/Realistics/cross_displacement.cpkl"
    # get_cross_displacement_data(realistics_path, saved_data_path)

    # saved_data_path = "datasets/Realistics/cross_subject_target-16.cpkl"
    # get_cross_subject_data(realistics_path, saved_data_path, target_id=16)

    saved_data_path = "datasets/Realistics/user_data-01.cpkl"
    get_user_data(realistics_path, saved_data_path, target_id=1)

    # saved_data_path = "datasets/Realistics/cross_multi_subjects-01.cpkl"
    # get_cross_mulit_subjects_data(realistics_path, saved_data_path, target_id=1)