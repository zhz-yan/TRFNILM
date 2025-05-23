import numpy as np
import soundfile as sf
import os

import torch
from tqdm import tqdm
from .transform import *

wanted_appl = ["Drill", "Fan", "Grinder", "Hair", "Hedge", "Lamp", "Sander", "Saw", "Vacuum"]
# wanted_appl = [0, 1, 2, 3, 4, 5, 6, 7, 8]

def align_voltage_current(i, v, NS):
    """ align voltage and current
    """
    current, voltage = np.copy(i), np.copy(v)
    zc = get_zero_crossing(voltage, NS)[1:]  

    for j in range(2, len(zc) - 2): 
        ts = zc[-j] - zc[-(j + 2)]
        ic = zero_crossings(current[zc[-(j + 2)]:zc[-j]])
        if ts == NS and len(ic) >= 2:
            c, v = current[zc[-(j + 2)]:zc[-j]], voltage[zc[-(j + 2)]:zc[-j]]
            break
        if ts > NS // 2 and len(ic) >= 2:
            c, v = current[zc[-(j + 4)]:zc[-j]], voltage[zc[-(j + 4)]:zc[-j]]
            break
    return c[:NS], v[:NS]


def get_cool_data(path):
    """ get the meta data and label

    Arguments:
        path {string} -- (Default: "D:/datasets/nilm/COOLL/")

    Returns:
        Meta {dictionary: (840,)} -- <key {int}: id, value {list}: meta data >
        labels {dictionary: (840)} -- <key {int}: id, value {string}: appliance type>
    """
    labels = {}
    delays = {}
    Meta = {}  # e.g., Meta = {id: [tbna0, act1, sofa1, ad1, tbna1, act2, sofa2, ad2, tbna2], ...}
    with os.scandir(path) as entries:  
        for entry in entries:
            if entry.is_dir():
                folder_name = entry.name
                with os.scandir(entry.path) as files:
                    for f in files:  # e.g., f : <DirEntry 'scenario1_1.txt'>
                        if f.is_file():
                            f_id = f.name.strip().split("_")[-1].strip().split(".")[0]
                            # if f_id=="141":
                            #     continue

                            if folder_name == "configs":  
                                meta_head = []
                                meta_data = []
                                with open(f.path, 'r') as l:
                                    line = l.readline()
                                    while line != '':  # The EOF char is an empty string
                                        if not line.lstrip().startswith('#'):
                                            meta = line.strip().split("=")

                                            meta_data.append(int(meta[-1]))
                                            meta_head.append(meta[0])
                                        line = l.readline()
                                    l.close()
                                    Meta[int(f_id)] = meta_data
                            if folder_name == "label":
                                with open(f.path, 'r') as l:
                                    line = l.readline()
                                    while line != '':  # The EOF char is an empty string
                                        app_id = line.strip().split("_")
                                        name = app_id[0].strip().split(":")
                                        labels[int(name[0])] = name[1].strip()
                                        delays[int(name[0])] = app_id[-1]
                                        # print(app_id[-1])
                                        line = l.readline()

    Meta[0] = meta_head  
    return Meta, labels


def get_cool_feature(path_dir):
    """ get cooll features

    Arguments:

    Returns:
        final_currents {ndarray: (839, Ts)} -- current data
        final_voltages {ndarray: (839, Ts)} -- voltage data
        final_labels {ndarray: (839, )} -- appliance type label

    """
    print("Load data")
    Meta, labels = get_cool_data(path_dir)
    Ts = int(100e3 / 50)  # the numbers of samples per periods
    cycles = Ts * 20  # total numbers of samples, 40000
    final_currents = []
    final_voltages = []
    final_labels = []

    files_id = 0
    with tqdm(total=len(Meta)) as pbar:
        for id in list(Meta.keys()):
            if id in [0, 141]:
                continue

            index_on = Meta[id][0] * int(1e2) + Meta[id][3]  # event on time (tbna0 * 100 + ad1)
            # print(id)
            if Meta[id][4] > 4000:  # if tbna1 > 4000
                start = 4000
            else:
                start = Meta[id][4]  # start = tbna1
            index_of = start * int(1e2) + Meta[id][7] + Meta[id][8] * int(
                1e2)  # event off time (tbna1 * 100 + ad2 + tbna2 *100)
            # i = {ndarray: (599999,)}, fs = {int} : 100 kHz
            i, fs = sf.read(path_dir + "current/scenarioC1_{}.flac".format(id))
            # v = {ndarray: (N,)}, fs = {int} : 100 kHz
            v, fs = sf.read(path_dir + "voltage/scenarioV1_{}.flac".format(id))

            c_on, v_on = i[index_on - cycles:index_on + cycles], v[index_on - cycles:index_on + cycles]
            # c_of,v_of = v[index_of-Ts*5:index_of+Ts*5], voltage[id][index_of-Ts*5:index_of+Ts*5]

            I, V = align_voltage_current(c_on, v_on, Ts)
            final_currents += [I]
            # final_currents+=[I_of]

            final_voltages += [V]
            # final_voltages+=[V_of]
            final_labels += [labels[id]]
            # final_labels+=[labels[id]]
            pbar.set_description('processed: %d' % (1 + files_id))
            pbar.update(1)
            files_id += 1
        pbar.close()

    print(f"labels:{len(final_labels)}")
    print(f"current:{len(final_currents)}")
    print(f"voltage:{len(final_voltages)}")

    return np.array(final_currents), np.array(final_voltages), np.array(final_labels)


def generate_image_label_pair(label, images):
    """ get cooll features

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


def get_train_test_leave_out_cooll(dataset, n=8):
    """ train test leave out cooll

    """
    houses = dict([(key, []) for key in range(n)])
    houses_ids = dict([(key, []) for key in range(n)])
    for name in wanted_appl:
        
        ids = np.array(range(len(dataset[name]))) // 20  
        ids = [i if i < n else i % n for i in ids] 

        for i in np.unique(ids):
            arr = list(range(n))  # arr: [0, 1, 2, 3, 4, 5, 6, 7]
            np.random.shuffle(arr)
            j = 0
            while True:
                if name in houses[arr[j]]:  
                    j += 1
                else: 
                    houses[arr[j]].append(name)  
                    houses_ids[arr[j]].append(i)  
                    break
    # houses_ids
    """ 
    for key, item in houses_ids.items():
        houses_ids[key]=list(np.unique(item))
    """

    train_set = []
    test_set = []
    index = 0

    for h, hi in zip(list(houses.values()), list(houses_ids.values())):
        test = {}
        train = {}
        # print(index)

        test_names = [i + str(j) for (i, j) in zip(h, hi)]
        for name in list(dataset.keys()):  # appliance type, e.g., name = {str} "Drill"
            ids = np.array(range(len(dataset[name]))) // 20  

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


def get_train_test_data(train_set, test_set, idx=0):
    """ get train test data

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
    mapping = {}
    id = 0

    for key, items in train.items():
        for i in range(len(items)):  #
            train_X.append(items[i])
            train_y.append(id)
        mapping[list(train.keys())[id]] = id  
        id += 1

    test_X = []
    test_y = []
    id = 0

    for key, items in test.items():
        for i in range(len(items)):
            test_X.append(items[i])
            test_y.append(mapping[list(test.keys())[id]])
        id += 1

    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)


# for online learning
