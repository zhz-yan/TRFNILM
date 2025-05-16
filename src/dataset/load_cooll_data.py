
from .transform import *

wanted_appl = ["Drill", "Fan", "Grinder", "Hair", "Hedge", "Lamp", "Sander", "Saw", "Vacuum"]
# wanted_appl = [0, 1, 2, 3, 4, 5, 6, 7, 8]


def generate_image_label_pair(label, images):

    dataset = {}
    for name in wanted_appl:
        index = np.where(label == name)[0]
        dataset[name] = images[index]

    return dataset


def get_train_test_leave_out_cooll(dataset, n=8):


    houses = dict([(key, []) for key in range(n)])
    houses_ids = dict([(key, []) for key in range(n)])
    for name in wanted_appl:

        ids = np.array(range(len(dataset[name]))) // 20
        ids = [i if i < n else i % n for i in ids]  #

        for i in np.unique(ids):
            arr = list(range(n))  # arr: [0, 1, 2, 3, 4, 5, 6, 7]
            np.random.shuffle(arr)
            j = 0
            while True:
                if name in houses[arr[j]]:
                    j += 1
                else:  # 若不存在
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
        for name in list(dataset.keys()):  # 遍历每个appliance type, e.g., name = {str} "Drill"
            ids = np.array(range(len(dataset[name]))) // 20  # 给appliance type 分配house id, e.g., ids = {ndarray:(120,)}

            for i in range(len(dataset[name])):  # 遍历每个appliance type 的instances的数组长度
                if name + str(ids[i]) in test_names:  # 如果 appliance type + house_id 存在于 test_names
                    if name not in test:  # 如果test{} 还没有该appliance type
                        test[name] = []
                    test[name].append(dataset[name][i])  # 给该test{} 添加相应的images数据
                elif name in wanted_appl:  # 否则制作训练集
                    if name not in train:
                        train[name] = []
                    train[name].append(dataset[name][i])
        test_set.append(test)
        train_set.append(train)
        index += 1

    return train_set, test_set


def get_train_test_data(train_set, test_set, idx=0):


    train = train_set[idx]
    test = test_set[idx]
    train_X = []
    train_y = []
    mapping = {}
    id = 0

    for key, items in train.items():
        for i in range(len(items)):
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
