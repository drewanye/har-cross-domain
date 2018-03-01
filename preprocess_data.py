__author__ = 'Zhenan Ye'

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
import cPickle as cp
import utils
import zipfile
from io import BytesIO
from sklearn import utils as skutils
import data_utils


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
    norm_young_data = (source_data-mean)/std
    norm_old_data = (target_data-mean)/std
    return norm_young_data, norm_old_data

def preprocess_data(dataset_path, saved_file_name):
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
    # normalize data
    norm_young_x_data, norm_old_x_data = normalize_data(young_x_data, old_x_data)
    num_labels = 17
    # data from the young as source domain, data from the old as target domain
    young_y_data = utils.one_hot(young_y_data[:, 0]-1, num_labels)
    old_y_data = utils.one_hot(old_y_data[:, 0]-1, num_labels)
    # shuffle data
    norm_young_x_data, young_y_data = skutils.shuffle(norm_young_x_data, young_y_data, random_state=0)
    norm_old_x_data, old_y_data = skutils.shuffle(norm_old_x_data, old_y_data, random_state=0)
    print("the size of source domain data: {} | the size of target domain data: {}".format(
        young_x_data.shape, old_x_data.shape))
    with open(os.path.join('datasets', saved_file_name), 'wb') as f:
        cp.dump(((norm_young_x_data, young_y_data), (norm_old_x_data, old_y_data)), f, protocol=cp.HIGHEST_PROTOCOL)


dataset_path = 'datasets'
realistics_path = os.path.join(dataset_path, 'realistics.zip')
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


def get_data(data_path, saved_file_name):
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
            import pdb;pdb.set_trace()
            if 'ideal' in filename:
                print('... file {} -> source data'.format(filename))
                source_x = np.vstack((source_x, data_x))
                source_y = np.concatenate((source_y, data_y))
            if 'self' in filename:
                print('... file {} -> source data'.format(filename))
                source_x = np.vstack((source_x, data_x))
                source_y = np.concatenate((source_y, data_y))
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

    # import pdb; pdb.set_trace()
    with open(os.path.join('datasets', saved_file_name), 'wb') as f:
        cp.dump(((source_x, source_y), (target_x, target_y)), f,
                protocol=cp.HIGHEST_PROTOCOL)

def convert_seq_data(data_path):
    '''
    [n, channels] -> [m, seq_length, channels]
    :param data_path:
    :return:
    '''
    with open(data_path, 'r') as f:
        source_data, target_data = cp.load(f)
        source_x, source_y = source_data
        target_x, target_y = target_data

    seq_length = 300
    channels = source_x.shape[1]

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

    def one_hot(y_, n_values):
        # y_ = y_.reshape(len(y_))
        # n_values = int(np.max(y_))+1
        return np.eye(n_values)[np.array(y_, dtype=np.int32)]

    seq_s_x, seq_s_y = slide_window(source_x, source_y)
    seq_s_y = [define_label(i) for i in seq_s_y]
    seq_s_y = np.asarray(seq_s_y)
    seq_t_x, seq_t_y = slide_window(target_x, target_y)
    seq_t_y = [define_label(i) for i in seq_t_y]
    seq_t_y = np.asarray(seq_t_y)


    # normalize
    seq_s_x, seq_t_x = normalize_data(seq_s_x, seq_t_x)
    source_x_data, source_y_data = skutils.shuffle(seq_s_x, seq_s_y, random_state=0)
    target_x_data, target_y_data = skutils.shuffle(seq_t_x, seq_t_y, random_state=0)

    source_y_data = one_hot(source_y_data-1, 33)
    target_y_data = one_hot(target_y_data-1, 33)

    print("the size of source domain data: {} | the size of target domain data: {}".format(
        source_x_data.shape, target_x_data.shape))

    with open(os.path.join('datasets', 'seq_realistics.cp'), 'wb') as f:
        cp.dump(((source_x_data, source_y_data), (target_x_data, target_y_data)), file=f)

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



if __name__=='__main__':
    # saved_file_name = 'all_UbiMiB.cp'
    # saved_file_name = 'normalized_UbiMiB.cp'
    # dataset_path = "datasets/UniMiB-SHAR/data"
    # preprocess_data(dataset_path, saved_file_name)

    saved_file_name = 'raw_realistics.cp'
    dataset_path = "datasets/realistics.zip"
    # get_data(dataset_path, saved_file_name)
    # convert_seq_data(os.path.join('datasets', saved_file_name))

    saved_data_path = "datasets/Realistics/cross_subject_target-16.cpkl"
    get_cross_subject_data(dataset_path, saved_data_path, target_id=16)