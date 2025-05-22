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
        self.path = path  # "D:/datasets/nilm/plaid/PLAID/"
        self.progress = progress  # True
        self.npts = npts  #  10000
        self.width = width
        self.sampling_frequency = fs  #  30 kHz
        self.mains_frequency = f0  #  50 Hz
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
            house_label (list):         
            house_Ids(dictionary), size = (64,)
            appliance_label (list):   
            appliance_Ids (dictionary): 
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
                i - 1 for i, j in enumerate(self.appliance_types, start=1) if j == t]  
            appliance_label[appliance_Ids[t]] = ii 
            Mapping[ii] = t  
        for (ii, t) in enumerate(self.house):
            house_Ids[t] = [i - 1 for i,
            j in enumerate(self.households, start=1) if j == t] 
            house_label[house_Ids[t]] = ii + 1  
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
                i - 1 for i, j in enumerate(self.appliance_types, start=1) if j == t]  
            appliance_label[appliance_Ids[t]] = ii  
            Mapping[ii] = t 
        for (ii, t) in enumerate(self.house):
            house_Ids[t] = [i - 1 for i,
            j in enumerate(self.households, start=1) if j == t]  
            house_label[house_Ids[t]] = ii + 1  
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
            tempI = np.sum(np.reshape(data[ind]['current'], [NP, NS]), 0) / NP  
            tempV = np.sum(np.reshape(data[ind]['voltage'], [NP, NS]), 0) / NP  
            # align current to make all samples start from 0 and goes up
            ix = np.argsort(np.abs(tempI))  
            j = 0
            while True:
                if ix[j] < 499 and tempI[ix[j] + 1] > tempI[
                    ix[j]]: 
                    real_ix = ix[j]
                    break
                else:  
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

    ix_train = [j for j in range(len(house_label)) if house_label[j] not in test_houses]
    ix_test = [j for j in range(len(house_label)) if house_label[j] in test_houses]

    ytrain = label[ix_train]
    ix_test = [j for j in ix_test if label[j] in ytrain]

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

    unique_houses = np.unique(house_label_test)

    for house in unique_houses:
        house_indices = np.where(house_label_test == house)[0]
        house_X = X_test[house_indices]
        house_y = y_test[house_indices]
        house_labels = house_label_test[house_indices]

        unique_labels = np.unique(house_y)
        for label in unique_labels:
            
            label_indices = np.where(house_y == label)[0]
            label_X = house_X[label_indices]
            label_y = house_y[label_indices]
            label_house_labels = house_labels[label_indices]

            if len(label_indices) <= k:
                tune_set.extend(label_X)
                tune_labels.extend(label_y)
                tune_house_labels.extend(label_house_labels)
            else:
                selected_indices = random.sample(range(len(label_indices)), k)
                tune_set.extend(label_X[selected_indices])
                tune_labels.extend(label_y[selected_indices])
                tune_house_labels.extend(label_house_labels[selected_indices])

                remaining_indices = [i for i in range(len(label_indices)) if i not in selected_indices]
                remaining_X_test.extend(label_X[remaining_indices])
                remaining_y_test.extend(label_y[remaining_indices])
                remaining_house_label_test.extend(label_house_labels[remaining_indices])

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
        house_indices = np.where(house_label_test == house)[0]

        num_tune_samples = max(1, int(len(house_indices) * percentage))

        tune_indices = np.random.choice(house_indices, num_tune_samples, replace=False)
        remaining_indices = list(set(house_indices) - set(tune_indices))

        tune_X.append(X_test[tune_indices])
        tune_y.append(y_test[tune_indices])
        tune_house_labels.append(house_label_test[tune_indices])

        remaining_X_test.append(X_test[remaining_indices])
        remaining_y_test.append(y_test[remaining_indices])
        remaining_house_labels.append(house_label_test[remaining_indices])

    tune_X = np.vstack(tune_X)
    tune_y = np.concatenate(tune_y)
    tune_house_labels = np.concatenate(tune_house_labels)
    remaining_X_test = np.vstack(remaining_X_test)
    remaining_y_test = np.concatenate(remaining_y_test)
    remaining_house_labels = np.concatenate(remaining_house_labels)

    return tune_X, tune_y, tune_house_labels, remaining_X_test, remaining_y_test, remaining_house_labels
