__author__ = 'Zhenan Ye'

import numpy as np
import matplotlib.pyplot as plt
import os
import cPickle as cpkl
import utils
import zipfile
from io import BytesIO
from sklearn import utils as skutils
import data_utils

Num_Classes = 8
Seq_Length = 300
Channels = 63
Subjects = 15

Sensors = {'acc': 'acc', 'gyr': 'Gyroscope', 'mag': 'MagneticField'}
Activities = {'climbingdown': 1, 'climbingup': 2, 'jumping': 3, 'lying': 4,
              'running': 5, 'sitting': 6, 'standing': 7, 'walking': 8}
Positions = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']

def concat(array):
    '''
    :param arrays: # 7*3, [*, 3]
            #                #      acc     #    gyr      #     mag
            #                #   (x, y, z)  #  (x, y, z)  #   (x, y, z)
            #    'chest'     #
            #    'forearm'   #
            #    'head'      #
            #    'shin'      #
            #    'thigh'     #
            #    'upperarm'  #
            #    'waist'     #
    :return:
            chest_acc(x, y, z), chest_gyr(x, y, z), chest_mag(x, y, z), forarm_acc, forearm_gyr, forarm_mag, ... , waist_acc, waist_gyr, waist_mag
    '''
    tmp = []
    rows = len(array)
    for i in range(rows):
        tmp.append(np.concatenate(array[i], axis=1))
    return np.concatenate(tmp, axis=1)

def slide_window(x, seq_length, channels):
    '''
    slide window without overlapping
    :param x:
    :param y:
    :return:
    '''
    num_x = x.shape[0] / seq_length * seq_length
    seq_x = np.reshape(x[:num_x], newshape=(-1, seq_length, channels))

    return seq_x

def truncate_and_transpose(array):

    min_value = array[0][0].shape[0]
    for a in array:
        for b in a:
            if min_value>b.shape[0]:
                min_value = b.shape[0]
    print "min value: {}".format(min_value)

    # transpose and truncate
    results = []
    rows = len(array)
    colums = len(array[0])
    for j in range(colums):
        tmp = []
        for i in range(rows):
            tmp.append(array[i][j][:min_value])
        results.append(tmp)
    return results

def generate_data(dataset_path, saved_path):
    with zipfile.ZipFile(dataset_path) as zf:

        for i in range(2, 3):
            data_x = np.empty((0, Channels))
            data_y = np.empty((0))
            for act in Activities:
                data_s = []
                for sen in Sensors:
                    file_path = 'proband{}/data/{}_{}_csv.zip'.format(i, sen, act)
                    print(" ... file path: {}".format(file_path))
                    zfiledata = BytesIO(zf.read(file_path))
                    with zipfile.ZipFile(zfiledata) as zf2:
                        data_p = []
                        for pos in Positions:
                            file_name = '{}_{}_{}.csv'.format(Sensors[sen], act, pos)
                            if file_name not in zf2.namelist():
                                file_name = '{}_{}_2_{}.csv'.format(Sensors[sen], act, pos)
                            print("  ... file name: {}".format(file_name))
                            data = np.genfromtxt(BytesIO(zf2.read(file_name)), delimiter=',', dtype=np.float32)
                            data_p.append(data[1:, -3:])
                            # data_x.append(slide_window(data[1:, -3:], Seq_Length, 3))
                    data_s.append(data_p)
                # 3*7
                #       'chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist'
                # acc
                # gyr
                # mag
                # [x, 7*9]
                matrix = truncate_and_transpose(data_s)
                data_a = concat(matrix)
                y = np.tile([0], data_a.shape[0])+Activities[act]
                data_x = np.concatenate((data_a, data_x))
                data_y = np.concatenate((y, data_y))
            with open(os.path.join(saved_path, 'subject{}.ckpl'.format(i)), 'wb') as f:
                import pdb;pdb.set_trace()
                print("saving subject {}".format(i))
                print("data size: X {}, y: {}".format(data_x.shape, data_y.shape))
                cpkl.dump([data_x, data_y], f)
                print "Completed!!!"

def get_cross_subject_data(data_path, saved_data_path, target_id):
    files = ['subject1.ckpl', 'subject3.ckpl', 'subject5.ckpl', 'subject8.ckpl', 'subject9.ckpl',
             'subject10.ckpl', 'subject11.ckpl', 'subject12.ckpl', 'subject13.ckpl', 'subject15.ckpl']
    # other subjects as source data, subject 9 as target data
    source_x = np.empty((0, Channels))
    source_y = np.empty((0))
    target_x = np.empty((0, Channels))
    target_y = np.empty((0))

    target_file = 'subject{}.ckpl'.format(target_id)
    for filename in files:
        with open(os.path.join(data_path, filename), 'r') as f:
            data_x, data_y = cpkl.load(f)
        if filename == target_file:
            print('... file {} -> target data'.format(filename))
            target_x = np.vstack((target_x, data_x))
            target_y = np.concatenate((target_y, data_y))
        else:
            print('... file {} -> source data'.format(filename))
            source_x = np.vstack((source_x, data_x))
            source_y = np.concatenate((source_y, data_y))

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
        cpkl.dump(((source_x, source_y), (target_x, target_y)), f)

def get_cross_placement_data(data_path, saved_data_path):
    files = ['subject1.ckpl', 'subject3.ckpl', 'subject5.ckpl', 'subject8.ckpl', 'subject9.ckpl',
             'subject10.ckpl', 'subject11.ckpl', 'subject12.ckpl', 'subject13.ckpl', 'subject15.ckpl']
    all_data_x = np.empty((0, Channels))
    all_data_y = np.empty((0))
    for filename in files:
        with open(os.path.join(data_path, filename), 'r') as f:
            print("...read file: {}".format(filename))
            data_x, data_y = cpkl.load(f)
            all_data_x = np.vstack((data_x, all_data_x))
            all_data_y = np.concatenate((data_y, all_data_y))
    # # upperarm -> forearm
    # print("upperarm -> forearm")
    # source_x = all_data_x[:, 45:54]
    # source_y = all_data_y
    # target_x = all_data_x[:, 9:18]
    # target_y = all_data_y

    # # thigh -> shin
    # print("thigh -> shin")
    # source_x = all_data_x[:, 36:45]
    # source_y = all_data_y
    # target_x = all_data_x[:, 27:36]
    # target_y = all_data_y

    # # forearm -> shin
    # print("forearm -> shin")
    # source_x = all_data_x[:, 9:18]
    # source_y = all_data_y
    # target_x = all_data_x[:, 27:36]
    # target_y = all_data_y

    # chest -> waist
    print("chest -> waist")
    source_x = all_data_x[:, 0:9]
    source_y = all_data_y
    target_x = all_data_x[:, 54:63]
    target_y = all_data_y

    # # (chest, forearm, shin) -> (waist, upperarm, thigh)
    # print("(chest, forearm, shin) -> (waist, upperarm, thigh)")
    # source_placements = (all_data_x[:, 0:9], all_data_x[:, 9:18], all_data_x[:, 27:36])
    # target_placements = (all_data_x[:, 54:63], all_data_x[:, 45:54], all_data_x[:, 36:45])
    # source_x = np.concatenate(source_placements, axis=1)
    # source_y = all_data_y
    # target_x = np.concatenate(target_placements, axis=1)
    # target_y = all_data_y

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
        cpkl.dump(((source_x, source_y), (target_x, target_y)), f)

if __name__ == '__main__':
    dataset_path = os.path.join('datasets', 'RealWorld')
    realistics_path = os.path.join(dataset_path, 'realworld2016_dataset.zip')
    # generate_data(realistics_path, dataset_path)

    # get cross subject data
    # target_id = 5
    # saved_data_path = os.path.join(dataset_path, 'cross_subject_{}.cpkl'.format(target_id))
    # get_cross_subject_data(dataset_path, saved_data_path, target_id=target_id)

    # get cross placement data: upperarm -> forearm
    # saved_data_path = os.path.join(dataset_path, 'cross_placement_upperarm_forearm.cpkl')
    # saved_data_path = os.path.join(dataset_path, 'cross_placement_thigh_shin.cpkl')
    saved_data_path = os.path.join(dataset_path, 'cross_placement_chest_waist.cpkl')
    get_cross_placement_data(dataset_path, saved_data_path)