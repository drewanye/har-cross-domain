__author__ = 'Zhenan Ye'

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
import cPickle as cp
import utils

Labels = [
    'StandingUpFS',
    'StandingUpFL',
    'Walking',
    'Running',
    'GoingUpS',
    'Jumping',
    'GoingDownS',
    'LyingDownFS',
    'SittingDown',
    'FallingForw',
    'FallingRight',
    'FallingBack',
    'HittingObstacle',
    'FallingWithPS',
    'FallingBackSC',
    'Syncope',
    'FallingLeft',
]

# age
subject_properties = {'2':[54], '3':[49], '7':[60], '8':[55]}
dataset_path = "datasets/UniMiB-SHAR/data"

def count_samples(data, suject_index, samples_info):
    num_sum = 0
    for i in Labels:
        print("...{}".format(i))
        d = data[i][0, 0]
        num_trials = d.shape[0]
        num_trials_sample = 0
        for j in range(num_trials):
            s = d[j, 0].shape[1]
            print(" ...trials {}, samples {}".format(j, s))
            num_trials_sample += s
        samples_info[suject_index, Labels.index(i)] = num_trials_sample
        num_sum += num_trials_sample
    return num_sum, samples_info

def magnitude(seg):
    return np.sqrt(seg[0, :] * seg[0, :] + seg[1, :] * seg[1, :] + seg[2, :] * seg[2, :])

def show_samples(full_data):
    data1 = full_data[0, 0]['Walking']
    # get different trials
    trails = data1[0, 0]
    # get one trail [6, 1863]
    trail1 = trails[0, 0]
    trail2 = trails[1, 0]
    seg = trail1[:, 0:151]
    fig, axs = plt.subplots(4)
    fig.tight_layout()
    axs[0].plot(seg[4, :], seg[0, :])
    axs[0].set_title('X accelerator')
    axs[1].plot(seg[4, :], seg[1, :])
    axs[1].set_title('Y accelerator')
    axs[2].plot(seg[4, :], seg[2, :])
    axs[2].set_title('Z accelerator')
    axs[3].plot(seg[4, :], magnitude(seg))
    axs[3].set_title('magnitude')
    plt.show()
    fig.savefig('diagram/sample.png')

def full_data():
    dataset_path = "datasets/UniMiB-SHAR/data/full_data.mat"
    mat_contents = sio.loadmat(dataset_path)
    samples_info = np.zeros([30, 17], dtype=np.int32)
    all_samples = 0
    num_subjects = 30
    full_data = mat_contents['full_data']
    for i in range(30):
        data = full_data[i, 0]
        num_samples, samples_info = count_samples(data, i, samples_info)
        print("subject {} samples {}\n".format(i, num_samples))
        all_samples += num_samples
    print("all samples {}".format(all_samples))
    print(samples_info)
    print(np.sum(samples_info, axis=0))

def show_accs_in_window(data, file_name):
    fig, axs = plt.subplots(3)
    plt.tight_layout()
    x = np.arange(0, 3.02, 0.02)
    axs[0].plot(x, data[0:151])
    axs[0].set_title('x_acc')
    axs[1].plot(x, data[151:302])
    axs[1].set_title('y_acc')
    axs[2].plot(x, data[302:])
    axs[2].set_title('z_acc')
    diagram_path = "diagram"
    fig.savefig(os.path.join(diagram_path,file_name))

def show_seg_data():
    x_data_path = os.path.join(dataset_path, 'acc_data.mat')
    y_data_path = os.path.join(dataset_path, 'acc_labels.mat')
    x_data = sio.loadmat(x_data_path)
    x_data = x_data['acc_data']
    y_data_path = sio.loadmat(y_data_path)
    y_data_path = y_data_path['acc_labels']

    # # StandingUpFS subject_id:line 2:7, 8:29, 1:2, 22:108
    # show_accs_in_window(x_data[6, :], "StandingUpFS_{}".format(54))
    # show_accs_in_window(x_data[28, :], "StandingUpFS_{}".format(55))
    # show_accs_in_window(x_data[1, :], "StandingUpFS_{}".format(24))
    # show_accs_in_window(x_data[107, :], "StandingUpFS_{}".format(18))

    # Walking subject_id:line 2:441, 8:749, 1:385, 22:1596
    show_accs_in_window(x_data[440, :], "Walking_{}".format(54))
    show_accs_in_window(x_data[748, :], "Walking_{}".format(55))
    show_accs_in_window(x_data[384, :], "Walking_{}".format(24))
    show_accs_in_window(x_data[1595, :], "Walking_{}".format(18))

def preprocess_data():
    x_data_path = os.path.join(dataset_path, 'acc_data.mat')
    y_data_path = os.path.join(dataset_path, 'acc_labels.mat')
    x_data = sio.loadmat(x_data_path)
    x_data = x_data['acc_data']
    y_data = sio.loadmat(y_data_path)
    y_data = y_data['acc_labels']
    old_group = [2, 3, 7, 8]
    old_group_index = np.where(np.isin(y_data[:, 1], old_group))
    young_group_index = np.where(np.invert(np.isin(y_data[:, 1], old_group)))
    young_x_data = x_data[young_group_index]
    young_y_data = y_data[young_group_index]
    old_x_data = x_data[old_group_index]
    old_y_data = y_data[old_group_index]

    # reshape x data as (:, 151, 3)
    young_x_data = np.reshape(young_x_data, [-1, 151, 3])
    old_x_data = np.reshape(old_x_data, [-1, 151, 3])
    num_labels = 17
    # data from the young as source domain, data from the old as target domain
    young_y_data = utils.one_hot(young_y_data[:, 0]-1, num_labels)
    old_y_data = utils.one_hot(old_y_data[:, 0]-1, num_labels)
    print("the size of source domain data: {} | the size of target domain data: {}".format(
        young_x_data.shape, old_x_data.shape))
    with open('datasets/all_UbiMiB.cp', 'wb') as f:
        cp.dump(((young_x_data, young_y_data), (old_x_data, old_y_data)), f, protocol=cp.HIGHEST_PROTOCOL)


if __name__=='__main__':
    preprocess_data()

