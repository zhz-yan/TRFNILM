
from .transform import *

# wanted_appl = ['CFL', 'DrillingMachine', 'Fan', 'FlatIron', 'GameConsole', 'HairDryer', 'Iron', 'Kettle', 'LEDLight',
#                'LightBulb', 'Massage', 'Microwave', 'Mixer', 'Monitor', 'PowerSupply', 'ShoeWarmer', 'Shredder',
#                'SolderingIron', 'Toaster', 'VacuumCleaner', 'WaterHeater']
wanted_appl = ['Charger', 'Fan', 'GameConsole', 'HairDryer', 'Kettle', 'LEDLight',
               'LightBulb', 'Microwave', 'Mixer', 'PowerSupply', 'TV', 'Toaster']
# wanted_appl = [0,1,2,3,4,5,6,7,8,9,10,11]


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
        ids = np.array(range(len(dataset[name]))) // 10
        for i in np.unique(ids):
            arr = list(range(n))                         # idx of houses
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
    index = 0
    # h: house and its appliances; hi: house and its appliance's id
    for h, hi in zip(list(houses.values()), list(houses_ids.values())):
        test = {}
        train = {}

        test_names = [i + str(j) for (i, j) in zip(h, hi)]
        for name in list(dataset.keys()):                       # appliance_name
            ids = np.array(range(len(dataset[name]))) // 10     # appliance's id
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

    houses = dict([(key, []) for key in range(n)])
    houses_ids = dict([(key, []) for key in range(n)])

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

        for name, samples in test.items():
            if len(samples) >= k:
                fine_tune[name] = samples[:k]
                test[name] = samples[k:]
            else:
                fine_tune[name] = samples
                test[name] = []

        test_set.append(test)
        train_set.append(train)
        fine_tune_set.append(fine_tune)
        index += 1

    return train_set, test_set, fine_tune_set


def get_train_test_data(train_set, test_set, idx=0):

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
