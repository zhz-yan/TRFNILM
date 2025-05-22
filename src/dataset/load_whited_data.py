import numpy as np
import soundfile as sf
import os
from tqdm import tqdm
from .transform import *
import torch

# wanted_appl = ['CFL', 'DrillingMachine', 'Fan', 'FlatIron', 'GameConsole', 'HairDryer', 'Iron', 'Kettle', 'LEDLight',
#                'LightBulb', 'Massage', 'Microwave', 'Mixer', 'Monitor', 'PowerSupply', 'ShoeWarmer', 'Shredder',
#                'SolderingIron', 'Toaster', 'VacuumCleaner', 'WaterHeater']
wanted_appl = ['Charger', 'Fan', 'GameConsole', 'HairDryer', 'Kettle', 'LEDLight',
               'LightBulb', 'Microwave', 'Mixer', 'PowerSupply', 'TV', 'Toaster']
# wanted_appl = [0,1,2,3,4,5,6,7,8,9,10,11]


def get_whited_data(path_dir):
    """
    :param path_dir:
    :return: appliance label, measurement data, sacle_factor and mk
    """
    appliance_type = []
    appliance_label = []
    region_label=[]
    region = []
    data = []
    scale_factor = np.empty([3,2])  # (mk1, mk2, mk3) * (v, i)
    mk = []

    with os.scandir(path_dir) as entries:
        for entry in entries:
            if entry.is_file():
                file_name = entry.name
                ext = file_name.strip().split(".")[-1]

                if ext == "txt":
                    f = open(path_dir + file_name)
                    all_data = f.readlines()
                    scale_factor[0, 0] = all_data[1].split()[2].split(";")[0]
                    scale_factor[0, 1] = all_data[2].split()[2].split(";")[0]

                    scale_factor[1, 0] = all_data[5].split()[2].split(";")[0]
                    scale_factor[1, 1] = all_data[6].split()[2].split(";")[0]

                    scale_factor[2, 0] = all_data[9].split()[2].split(";")[0]
                    scale_factor[2, 1] = all_data[10].split()[2].split(";")[0]

                if ext == "flac":
                    names = file_name.strip().split(".")[0].strip().split("_")
                    appliance_type.append(names[0])
                    appliance_label.append(names[1])
                    region.append(names[2])
                    mk.append(names[3])

                    d, fs = sf.read(entry.path)
                    data.append(d)
                    region_label.append(int(list(names[2])[-1]))

    label = np.array(appliance_type)
    data = np.array(data)

    return label, data, scale_factor, mk, region


def denormalization(data, scale_factor, mk):
    """
    denormalize the voltage and current data
    :param data: data[:,:,0] is voltage, and data[:,:,1] is current
    :param scale_factor: scale_factor[:,0] is  voltage scale factor, and  scale_factor[:,0] is for current
    :param mk:
    :return:
    """
    # data[:,:,0] -- voltage
    # data[:,:,1] -- current
    #
    # scale_fator[:,1] - current scale factor
    print(data.shape)
    print(scale_factor.shape)
    n = len(data)
    for i in range(n):
        if mk[i] == "MK1":
            data[i, :, 0] *= scale_factor[0, 0]
            data[i, :, 1] *= scale_factor[0, 1]

        elif mk[i] == "MK2":
            data[i, :, 0] *= scale_factor[1, 0]
            data[i, :, 1] *= scale_factor[1, 1]

        else:  # MK3
            data[i, :, 0] *= scale_factor[2, 0]
            data[i, :, 1] *= scale_factor[2, 1]

    return data


# average m cycles of voltage and current
def get_trajectory(data, fs=44.1e3, f0=50):
    NS = int(fs / f0)
    NP = 20
    npts = int(NS * 20)
    n = len(data)
    I = np.empty([n, NS])
    V = np.empty([n, NS])
    files_id = 0

    with tqdm(total=n) as pbar:
        for ind in range(n):

            tempI = np.sum(np.reshape(data[ind][-npts:, 1], [NP, NS]), 0) / NP
            tempV = np.sum(np.reshape(data[ind][-npts:, 0], [NP, NS]), 0) / NP

            ix = np.argsort(np.abs(tempI))
            j = 0
            while True:
                if ix[j] < NS - 1 and tempI[ix[j] + 1] > tempI[ix[j]]:
                    real_ix = ix[j]
                    break
                else:
                    j += 1
            c = np.hstack([tempI[real_ix:], tempI[:real_ix]])
            v = np.hstack([tempV[real_ix:], tempV[:real_ix]])
            I[ind,] = c
            V[ind,] = v
            pbar.set_description('processed: %d' % (1 + files_id))
            pbar.update(1)
            files_id += 1
        pbar.close()

    return I, V


def get_whited_feature(path_dir):
    """ get white features

    Arguments:

    Returns:
        current {ndarray: (1339, Ts)} -- current data
        voltage {ndarray: (1339, Ts)} -- voltage data
        label {ndarray: (1339, )} -- appliance type label

    """

    label, data, scale_factor, mk, region = get_whited_data(path_dir)
    data = denormalization(data, scale_factor, mk)
    current, voltage = get_trajectory(data)  # 求平均电流
    # current, voltage, label = get_VI_trajectory(data, label, fs=44.1e3, f0=50)

    return current, voltage, label, region


def generate_dataset_whited(label, images):
    """ get whited features

    Arguments:
        label {ndarray: (839,)} -- appliance types
        images {ndarray: (839, 1, W, W)} -- input images

    Returns:
        data {dict:9} -- <key: appliance types, value: images>

    """
    dataset = {}
    for name in wanted_appl:
        index = np.where(label == name)[0]
        dataset[name] = images[index]

    return dataset


def get_train_test_leave_out_whited(dataset, n=9):

    houses = dict([(key, []) for key in range(n)])
    houses_ids = dict([(key, []) for key in range(n)])

    for name in wanted_appl:
        ids = np.array(range(len(dataset[name]))) // 10  # 每一个app分配给哪个house
        for i in np.unique(ids):                         # 第i个house
            arr = list(range(n))                         # idx of houses
            np.random.shuffle(arr)
            j = 0
            while True:
                if name in houses[arr[j]]:               # 选择某一个house，查看里面是否有设备 name
                    j += 1
                else:
                    houses[arr[j]].append(name)         # 给house[arr[j]]添加设备name
                    houses_ids[arr[j]].append(i)        # 设备name 对应的 ids
                    break

    train_set = []
    test_set = []
    index = 0
    # h: house and its appliances; hi: house and its appliance's id
    for h, hi in zip(list(houses.values()), list(houses_ids.values())):
        test = {}
        train = {}

        test_names = [i + str(j) for (i, j) in zip(h, hi)]
        for name in list(dataset.keys()):                       # appliance_name
            ids = np.array(range(len(dataset[name]))) // 10     # appliance的id(按house)
            for i in range(len(dataset[name])):
                if name + str(ids[i]) in test_names:
                    if name not in test:
                        test[name] = []
                    test[name].append(dataset[name][i])
                elif name in wanted_appl:
                    if name not in train:
                        train[name] = []
                    train[name].append(dataset[name][i])
        test_set.append(test)
        train_set.append(train)
        index += 1

    return train_set, test_set


def get_train_test_leave_out_whited_with_tune(dataset, n=9, k=3):
    """
    将数据按leave-k-house-out划分为训练集、测试集和微调集。

    Arguments:
        dataset {dict} -- 数据集，包含不同类别的用电负荷数据，每个类别是一个列表，列表包含了不同house的数据
        n {int} -- 总的house数量 (默认值: 9)
        k {int} -- 每个house的每个类别中用于fine-tune的样本数量 (默认值: 5)

    Returns:
        train_set {list: (n,)} -- 列表，每个元素是一个字典，包含每个house的训练数据
        test_set {list: (n,)} -- 列表，每个元素是一个字典，包含每个house的测试数据
        fine_tune_set {list: (n,)} -- 列表，每个元素是一个字典，包含每个house的fine-tune数据
    """

    houses = dict([(key, []) for key in range(n)])
    houses_ids = dict([(key, []) for key in range(n)])

    # 遍历所需的用电负荷类型，将每个类别的数据分配到不同的house
    for name in wanted_appl:
        ids = np.array(range(len(dataset[name]))) // 10
        for i in np.unique(ids):
            arr = list(range(n))
            np.random.shuffle(arr)
            j = 0
            while True:
                if name in houses[arr[j]]:
                    j += 1
                else:
                    houses[arr[j]].append(name)
                    houses_ids[arr[j]].append(i)
                    break

    train_set = []
    test_set = []
    fine_tune_set = []
    index = 0

    for h, hi in zip(list(houses.values()), list(houses_ids.values())):
        test = {}
        train = {}
        fine_tune = {}

        test_names = [i + j for (i, j) in zip(h, hi)]
        for name in list(dataset.keys()):
            ids = np.array(range(len(dataset[name]))) // 10
            for i in range(len(dataset[name])):
                if name + ids[i] in test_names:
                    if name not in test:
                        test[name] = []
                    test[name].append(dataset[name][i])
                elif name in wanted_appl:
                    if name not in train:
                        train[name] = []
                    train[name].append(dataset[name][i])

        # 从测试集中抽取k个样本作为fine-tune数据
        for name, samples in test.items():
            if len(samples) >= k:
                fine_tune[name] = samples[:k]
                test[name] = samples[k:]  # 剩下的数据保留在测试集中
            else:
                fine_tune[name] = samples  # 如果不足k个，则全部加入fine-tune集中
                test[name] = []  # 清空该类别的测试数据

        test_set.append(test)
        train_set.append(train)
        fine_tune_set.append(fine_tune)
        index += 1

    return train_set, test_set, fine_tune_set


def get_train_test_data(train_set, test_set, idx=0):
    """ get train test data

       Arguments:
           train_set {list: (n,)} -- 列表里的每个元素为 test {dic},对应house i里的负荷的数据，用于测试
           test_set {list: (n,)} --  列表里的每个元素为 train {dic}, 对应剩下n-1个house里的数据，用于训练
           idx = {int} -- house index for testing

       Returns:
           train_X = {Tensor: (n_train,1,W,W)} -- training data
           train_y = {Tensor: (n_train,)} -- training labels
           test_X = {Tensor: (n_test,1,W,W)} -- testing data
           test_y = {Tensor: (n_test,)} -- testing labels
    """
    train = train_set[idx]
    test = test_set[idx]

    train_X = []
    train_y = []

    for key, items in train.items():
        for i in range(len(items)):
            train_X.append(items[i])
            train_y.append(key)

    test_X = []
    test_y = []

    for key, items in test.items():
        for i in range(len(items)):
            test_X.append(items[i])
            test_y.append(key)

    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)


def get_train_test_data_exp1(train, test, mapping):

    train_X, train_y = [], []

    for key, items in train.items():
        for i in range(len(items)):  #
            train_X.append(items[i])
            train_y.append(mapping[key])

    test_X, test_y = [], []

    for key, items in test.items():
        for i in range(len(items)):
            test_X.append(items[i])
            test_y.append(mapping[key])

    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)
