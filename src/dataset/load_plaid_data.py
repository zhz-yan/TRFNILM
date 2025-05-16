import numpy as np
import random


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
