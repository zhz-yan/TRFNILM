import os
import numpy as np
import pickle
import random
import json
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt


class PLAID(object):

    def __init__(self, path, progress=True, width=50, npts=10000, fs=30000, f0=60):
        self.path = path  # 数据路径, "D:/datasets/nilm/plaid/PLAID/"
        self.progress = progress  # True
        self.npts = npts  # 采样点, 10000
        self.width = width
        self.sampling_frequency = fs  # 采样率, 30 kHz
        self.mains_frequency = f0  # 基波频率, 50 Hz
        self.get_meta_parameters()  # Meta data

    def clean_meta(self, ist):
        '''remove None elements in Meta Data '''
        clean_ist = ist.copy()  # {"appliance":{...}, "header"：{...}, "instances":{...}, "location":{...}}
        for k, v in ist.items():  # # e.g., k = "appliance", v = {...}
            if len(v) == 0:
                del clean_ist[k]
        return clean_ist

    def get_meta_data(self):
        """ 返回.json 文件
            M (dictionary):  {1:{'applinace':{...}, "header":{...}, "instances":{...}, "location":{...}}, 2:...}
        """
        with open(self.path + 'meta_2014.json') as data_file:
            meta1 = json.load(data_file)

        with open(self.path + 'meta_2017.json') as data_file:
            meta2 = json.load(data_file)

        M = {}

        # consider PLAID1 and 2 [meta1, meta2]
        for m in [meta1, meta2]:
            for app in m:
                M[int(app['id'])] = self.clean_meta(app['meta'])

        return M

    def get_meta_parameters(self):
        # applinace types of all instances
        Meta = self.get_meta_data()  # Meta (dictionary): .json files. size = (1793,)
        self.appliance_types = [x["appliance"]['type'] for x in Meta.values()]  # appliance type (list): size = (1793,)

        # unique appliance types
        self.appliances = list(set(self.appliance_types))  # unique appliance type (list): size = (12,)
        self.appliances.sort()

        self.data_IDs = list(Meta.keys())  # data id (list): size = (1793,)

        # households of appliances
        self.households = [x['header']['collection_time'] +
                           '_' + x['location'] for x in Meta.values()]  # houses (list): size = (1793,)
        #   unique households
        self.house = list(set(self.households))  # unique houses (list): size = (64,)
        self.house.sort()

        print(f'Number of appliances:{len(self.appliances)} \nNumber of households:{len(self.house)}\
             \nNumber of total measurements:{len(self.households)}')

    def get_data(self):
        """
        return:
            data (dictionary):   key is the data_id, value is a current and voltage data. size = (1793,)
            house_label (list):         把 self.households上的 house 转化为 house_id, 1-based
            house_Ids(dictionary):      每个房屋(houses)在house_label的上的index, 0-based, size = (64,)
            appliance_label (list):     把 self.appliance上的 appliance 转化为 appliance_id, 1-based
            appliance_Ids (dictionary): 每种设备(appliance type)在appliance_label的上的index, 0-based, size = (12,)
        """
        path = self.path + '2017/'  # csv file path
        last_offset = self.npts  # numbers of sampling points
        start = datetime.now()
        n = len(self.data_IDs)  # numbers of samples
        if n == 0:
            return {}
        else:
            data = {}
            for (i, ist_id) in enumerate(self.data_IDs, start=1):
                if self.progress and np.mod(i, np.ceil(n / 10)) == 0:
                    print('%d/%d (%2.0f%s) have been read...\t time consumed: %ds'
                          % (i, n, i / n * 100, '%', (datetime.now() - start).seconds))
                if last_offset == 0:
                    data[ist_id] = np.genfromtxt(path + str(ist_id) + '.csv', delimiter=',',
                                                 names='current,voltage', dtype=(float, float))
                else:
                    p = path + str(ist_id) + ".csv"
                    with open(p, 'rb') as f:
                        lines = f.readlines()
                        data[ist_id] = np.genfromtxt(lines[-last_offset:], delimiter=',',
                                                     names='current,voltage', dtype=(float, float))
                        f.close()

                    # p = subprocess.Popen(['tail', '-'+str(int(last_offset)), path+str(ist_id)+'.csv'],
                    #                      stdout=subprocess.PIPE)
                    # data[ist_id] = np.genfromtxt(p.stdout, delimiter=',', names='current,voltage',
                    #                              dtype=(float, float))
            print('%d/%d (%2.0f%s) have been read(Done!) \t time consumed: %ds'
                  % (n, n, 100, '%', (datetime.now() - start).seconds))

        appliance_Ids = {}
        house_Ids = {}
        Mapping = {}
        n = len(data)  # number of samples
        appliance_label = np.zeros(n, dtype='int')
        house_label = np.zeros(n, dtype='int')

        for (ii, t) in enumerate(self.appliances):
            appliance_Ids[t] = [
                i - 1 for i, j in enumerate(self.appliance_types, start=1) if j == t]  # 字典，每种设备在Types的上的索引, 0-based
            appliance_label[appliance_Ids[t]] = ii  # 列表，把type上的设备名转化为appliance_id
            Mapping[ii] = t  # 字典，appliance_id -> 设备 的映射
        for (ii, t) in enumerate(self.house):
            house_Ids[t] = [i - 1 for i,
            j in enumerate(self.households, start=1) if j == t]  # 字典，每个house在Locs的上的索引, 0-based
            house_label[house_Ids[t]] = ii + 1  # 列表，把Locs上的设备名转化为house_id, 1-based
        print('number of different appliances: %d' % len(self.appliances))
        print('number of different households: %d' % len(self.house))
        return data, house_label, house_Ids, appliance_label, appliance_Ids

    def get_house_id_data(self):

        appliance_Ids = {}
        house_Ids = {}
        Mapping = {}
        n = len(self.data_IDs)  # numbers of samples
        appliance_label = np.zeros(n, dtype='int')
        house_label = np.zeros(n, dtype='int')

        for (ii, t) in enumerate(self.appliances):
            appliance_Ids[t] = [
                i - 1 for i, j in enumerate(self.appliance_types, start=1) if j == t]  # 字典，每种设备在Types的上的索引, 0-based
            appliance_label[appliance_Ids[t]] = ii  # 列表，把type上的设备名转化为appliance_id
            Mapping[ii] = t  # 字典，appliance_id -> 设备 的映射
        for (ii, t) in enumerate(self.house):
            house_Ids[t] = [i - 1 for i,
            j in enumerate(self.households, start=1) if j == t]  # 字典，每个house在Locs的上的索引, 0-based
            house_label[house_Ids[t]] = ii + 1  # 列表，把Locs上的设备名转化为house_id, 1-based
        print('number of different appliances: %d' % len(self.appliances))
        print('number of different households: %d' % len(self.house))
        return house_label, house_Ids, appliance_label, appliance_Ids


    def get_features(self, data):
        """
        return:
            rep_I (np.ndarray): representative one period of steady state current. size = (N, 500)
            rep_V (np.ndarray): representative one period of steady state voltage. size = (N, 500)
        """
        # number of samples per period
        NS = int(self.sampling_frequency // self.mains_frequency)  # 30000 // 60 = 500
        NP = int(self.npts / NS)  # number of periods for npts            # 10000 / 500 = 20

        # calculate the representative one period of steady state
        # (mean of the aggregated signals over one cycle)
        n = len(data)
        rep_I = np.empty([n, NS])
        rep_V = np.empty([n, NS])
        for i in range(n):
            ind = list(data)[i]  # 取dictionary的ke
            tempI = np.sum(np.reshape(data[ind]['current'], [NP, NS]), 0) / NP  # 求一个周波的平均电流
            tempV = np.sum(np.reshape(data[ind]['voltage'], [NP, NS]), 0) / NP  # 求一个周波的平均电压
            # align current to make all samples start from 0 and goes up
            ix = np.argsort(np.abs(tempI))  # 经过abs, 按顺序找最小到最大值的位置
            j = 0
            while True:
                if ix[j] < 499 and tempI[ix[j] + 1] > tempI[
                    ix[j]]:  # tempI[ix[j]] 和 tempI[ix[j]+1] 均为正; 或 tempI[ix[j]]为负，tempI[ix[j]+1]为正
                    real_ix = ix[j]
                    break
                else:  # tempI[ix[j]]为正，tempI[ix[j]+1]为负; 或tempI[ix[j]] 和 tempI[ix[j]+1] 均为负
                    j += 1
            rep_I[i,] = np.hstack([tempI[real_ix:], tempI[:real_ix]])
            rep_V[i,] = np.hstack([tempV[real_ix:], tempV[:real_ix]])

            # rep_I[i, ] = rep_I[i, ] / max(rep_I[i,])               #  normalization
            # rep_V[i, ] = rep_V[i, ] / max(rep_V[i,])

        return rep_I, rep_V


def get_plaid_data(path, width=50):
    """
    return:
        current {np.ndarray: (N, 500)}: representative one period of steady state current.
        voltage {np.ndarray: (N, 500)}: representative one period of steady state voltage.
        appliance_label {list: (N,)}:   house_id, 1-based.
        house_label {list: (N,)}: appliance_id, 1-based.
    where N is the number of samples
    """
    plaid = PLAID(path=path)

    data, house_label, house_Ids, appliance_label, appliance_Ids = plaid.get_data()
    current, voltage = plaid.get_features(data)

    return current, voltage, appliance_label, house_label


def leave_one_house_out_plaid(house_label, label, input_feature, amount_houses_test):

    test_houses = random.sample(set(house_label), amount_houses_test)

    # 构建训练集和测试集索引
    ix_train = [j for j in range(len(house_label)) if house_label[j] not in test_houses]
    ix_test = [j for j in range(len(house_label)) if house_label[j] in test_houses]

    # 确保测试集的标签在训练集中存在
    ytrain = label[ix_train]
    ix_test = [j for j in ix_test if label[j] in ytrain]

    # 获取训练集和测试集的特征和标签
    Xtrain, Xtest = input_feature[ix_train], input_feature[ix_test]
    ytrain, ytest = label[ix_train], label[ix_test]
    house_label_test = np.array(house_label)[ix_test]

    return np.array(Xtrain), np.array(Xtest), np.array(ytrain), np.array(ytest), house_label_test


def split_tune_set(X_test, y_test, house_label_test, k):
    tune_set = []
    tune_labels = []
    tune_house_labels = []

    remaining_X_test = []
    remaining_y_test = []
    remaining_house_label_test = []

    # 获取唯一的house和label组合
    unique_houses = np.unique(house_label_test)

    for house in unique_houses:
        # 获取属于该house的样本索引
        house_indices = np.where(house_label_test == house)[0]
        house_X = X_test[house_indices]
        house_y = y_test[house_indices]
        house_labels = house_label_test[house_indices]

        # 按类别分组
        unique_labels = np.unique(house_y)
        for label in unique_labels:
            # 获取该house中属于该label的样本索引
            label_indices = np.where(house_y == label)[0]
            label_X = house_X[label_indices]
            label_y = house_y[label_indices]
            label_house_labels = house_labels[label_indices]

            # 如果样本数量小于k，全部取出；否则随机取k个
            if len(label_indices) <= k:
                tune_set.extend(label_X)
                tune_labels.extend(label_y)
                tune_house_labels.extend(label_house_labels)
            else:
                selected_indices = random.sample(range(len(label_indices)), k)
                tune_set.extend(label_X[selected_indices])
                tune_labels.extend(label_y[selected_indices])
                tune_house_labels.extend(label_house_labels[selected_indices])

                # 其余的样本作为剩余
                remaining_indices = [i for i in range(len(label_indices)) if i not in selected_indices]
                remaining_X_test.extend(label_X[remaining_indices])
                remaining_y_test.extend(label_y[remaining_indices])
                remaining_house_label_test.extend(label_house_labels[remaining_indices])

    # 转换为ndarray
    tune_set = np.array(tune_set)
    tune_labels = np.array(tune_labels)
    tune_house_labels = np.array(tune_house_labels)

    remaining_X_test = np.array(remaining_X_test)
    remaining_y_test = np.array(remaining_y_test)
    remaining_house_label_test = np.array(remaining_house_label_test)

    return tune_set, tune_labels, tune_house_labels, remaining_X_test, remaining_y_test, remaining_house_label_test


def split_tune_set_percentage(X_test, y_test, house_label_test, percentage):
    """
    Split test set into tune set and remaining test set based on percentage.

    Parameters:
    - X_test: ndarray, test set features
    - y_test: ndarray, test set labels
    - house_label_test: ndarray, test set house labels
    - percentage: float, percentage of samples to extract from each house

    Returns:
    - tune_X: ndarray, features of the tune set
    - tune_y: ndarray, labels of the tune set
    - tune_house_labels: ndarray, house labels of the tune set
    - remaining_X_test: ndarray, features of the remaining test set
    - remaining_y_test: ndarray, labels of the remaining test set
    - remaining_house_labels: ndarray, house labels of the remaining test set
    """
    tune_X, tune_y, tune_house_labels = [], [], []
    remaining_X_test, remaining_y_test, remaining_house_labels = [], [], []

    unique_houses = np.unique(house_label_test)

    for house in unique_houses:
        # 找到属于当前 house 的所有索引
        house_indices = np.where(house_label_test == house)[0]

        # 计算 10% 的样本数量
        num_tune_samples = max(1, int(len(house_indices) * percentage))

        # 随机选择 num_tune_samples 个索引作为 tune_set
        tune_indices = np.random.choice(house_indices, num_tune_samples, replace=False)
        remaining_indices = list(set(house_indices) - set(tune_indices))

        # 添加到 tune_set
        tune_X.append(X_test[tune_indices])
        tune_y.append(y_test[tune_indices])
        tune_house_labels.append(house_label_test[tune_indices])

        # 添加到 remaining_test_set
        remaining_X_test.append(X_test[remaining_indices])
        remaining_y_test.append(y_test[remaining_indices])
        remaining_house_labels.append(house_label_test[remaining_indices])

    # 合并所有 house 的数据
    tune_X = np.vstack(tune_X)
    tune_y = np.concatenate(tune_y)
    tune_house_labels = np.concatenate(tune_house_labels)
    remaining_X_test = np.vstack(remaining_X_test)
    remaining_y_test = np.concatenate(remaining_y_test)
    remaining_house_labels = np.concatenate(remaining_house_labels)

    return tune_X, tune_y, tune_house_labels, remaining_X_test, remaining_y_test, remaining_house_labels
