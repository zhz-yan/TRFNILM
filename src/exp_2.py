
import argparse
import os
import json

import numpy as np
import psutil

from load_feat import create_features
import time
import joblib

from utils.data_generator import *
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from dataset.load_cooll_data import *
from dataset.load_whited_data import *
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from dataset.load_plaid_data import leave_one_house_out_plaid, split_tune_set, split_tune_set_percentage
# from SER.ser import SER_RF
from SER.WeightedRF import WeightedRandomForest


def sample_from_large_categories(Xt, yt, n_samples=1):
    # houses = dict([(key, []) for key in range(n)])
    # houses_ids = dict([(key, []) for key in range(n)])
    wanted_appl = np.unique(yt)

    idx = []
    for name in wanted_appl:
        appl_idx = np.where(yt == name)[0]
        for i in range(0, len(appl_idx), 10):
            appl_idx_i = np.random.choice(appl_idx[i:i+10], n_samples, replace=False)
            idx.append(appl_idx_i)

    idx = np.array(idx).flatten()
    Xtune, ytune = Xt[idx], yt[idx]
    Xtune, ytune = Xtune.reshape(len(idx), -1), ytune.reshape(-1)

    # mask = np.ones(len(Xt), dtype=bool)
    # mask[idx] = False  # Set positions in idx to False

    Xtest, ytest = np.delete(Xt, idx, axis=0), np.delete(yt, idx, axis=0)
    # Xtest, Xtune_r, ytest, ytune_r
    return Xtest, Xtune, ytest, ytune


def exp_case2_transfer(args, input_feature, label, dataset, house_label=None, amount_houses_test=1):
    classes = list(np.unique(label))
    num_class = len(classes)
    process = psutil.Process()

    if dataset == "whited":
        data = generate_dataset_whited(label, input_feature)
        train_set, test_set = get_train_test_leave_out_whited(data, 9)
        n = len(train_set)

    if dataset == "cooll":
        data = generate_image_label_pair(label, input_feature)
        train_set, test_set = get_train_test_leave_out_cooll(data, 8)
        n = len(train_set)

    if dataset == "plaid":
        houses = np.unique(house_label)
        n = len(houses)

    # Number of Leave-one-house-out
    for i in range(n):
        if dataset == "plaid":
            Xtrain, Xtest, ytrain, ytest, house_label_test = leave_one_house_out_plaid(house_label, label, input_feature, amount_houses_test=1)

        if dataset in ["whited", "cooll"]:
            Xtrain, ytrain, Xtest, ytest = get_train_test_data(train_set, test_set, idx=i)

            # Xtrain, ytrain, Xtest, ytest, Xtune, ytune = get_train_test_tune_data(train_set, test_set, tune_set, idx=i)

        le = LabelEncoder()
        ytrain = le.fit_transform(ytrain)
        ytest = le.transform(ytest)

        Xtrain, X_val, ytrain, y_val = train_test_split(Xtrain, ytrain, test_size=0.2, stratify=ytrain)

        """for others """
        if dataset == "plaid":
            X_tune, y_tune, _, Xtest, ytest, _ = split_tune_set(Xtest, ytest, house_label_test, args.k_target)
        else:
            Xtest, X_tune, ytest, y_tune = sample_from_large_categories(Xtest, ytest, n_samples=args.k_target)

        # label encoding
        src_model = RandomForestClassifier(n_estimators=args.tree)

        src_model.fit(Xtrain, ytrain)
        tgt_model = WeightedRandomForest(src_model, n_update=0.8, w_new=0.8, original_ser=False)
        tgt_model.update_forest(X_tune, y_tune)

        yt_pred = tgt_model.predict(Xtest)

        results = {
            "Target": {
                "Accuracy": float(accuracy_score(ytest, yt_pred) * 100),
                "F1_macro": float(f1_score(ytest, yt_pred, average="macro") * 100),
                "Precision": float(precision_score(ytest, yt_pred, average="macro") * 100),
                "Recall": float(recall_score(ytest, yt_pred, average="macro") * 100),
                "Number of samples": len(ytest)
            }
        }

        file_path = f"results/{str(args.dataset)}/kt_{str(args.k_target)}_ori/WTRF_{str(i)}.json"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4, separators=(",", ": "))


parser = argparse.ArgumentParser("Training")
# Dataset
parser.add_argument('--dataset', type=str, default='whited', help="plaid, whited, cooll")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./log')
parser.add_argument('--pre_save', type=bool, default=True)
parser.add_argument('--k_source', type=int, default=0)
parser.add_argument('--k_target', type=int, default=1)
parser.add_argument('--exp_num', type=int, default=1)
parser.add_argument('--amount_houses_test', type=int, default=1)
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--tree', type=int, default=9, help="number of trees")
parser.add_argument('--n_update', type=int, default=0.8)


args = parser.parse_args()

if __name__ == '__main__':
    print("start...")

    print("Starting generalization experiment...")
    print(f"Loading {args.dataset} dataset...")

    data = np.load(f"data/{args.dataset}/X.npy")        # features
    labels = np.load(f"data/{args.dataset}/Y.npy")
    if args.dataset == "plaid":
        house_label = np.load(f"data/{args.dataset}/house_label.npy", allow_pickle=True)
    else:
        house_label = None

    # sample_from_large_categories(data_2, labels_2)
    # for num in range(0, 21):
    exp_case2_transfer(args=args, input_feature=data,
                       label=labels,
                       dataset=args.dataset,
                       house_label=house_label,
                       amount_houses_test=args.amount_houses_test)
        # args.exp_num += 1