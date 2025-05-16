
import numpy as np
from .ser import SER, get_node_distribution
from sklearn.metrics import accuracy_score


class WeightedRandomForest:
    def __init__(self, trained_forest, n_update=0.8, original_ser=True, bootstrap_=False,
           no_red_on_cl=False, cl_no_red=None, no_ext_on_cl=False, cl_no_ext=None,
           ext_cond=False, leaf_loss_quantify=False, leaf_loss_threshold=0.9, w_new=0.8):

        self.T = trained_forest.estimators_  # Load trees from the existing trained RandomForestClassifier
        self.n_update = int(trained_forest.n_estimators * n_update)  # Number of trees to update each iteration
        self.w_new = w_new  # Weight for new trees
        self.w_old = 1 - w_new  # Weight for old trees
        self.original_ser = original_ser
        self.bootstrap_ = bootstrap_
        self.no_red_on_cl = no_red_on_cl
        self.cl_no_red = cl_no_red
        self.no_ext_on_cl = no_ext_on_cl
        self.cl_no_ext = cl_no_ext
        self.ext_cond = ext_cond
        self.leaf_loss_quantify = leaf_loss_quantify
        self.leaf_loss_threshold = leaf_loss_threshold

    def update_forest(self, X_target, y_target):
        # Randomly select `n_update` trees for modification
        T_new_indices = np.random.choice(len(self.T), self.n_update, replace=False)
        T_old_indices = [i for i in range(len(self.T)) if i not in T_new_indices]

        # Update selected trees and replace them in the forest
        # for idx in T_new_indices:
        #     n_r = np.random.poisson(1)  # Number of times to update this tree
        #     for _ in range(n_r):
        #
        #         self.T[idx] = self.SER(self.T[idx], X_new, y_new)  # Update tree using SER strategy
        for i, dtree in enumerate(T_new_indices):
            # n_r = np.random.poisson(1)  # Number of times to update this tree
            # for _ in range(n_r):
            root_source_values = None
            coeffs = None
            Nkmin = None
            if self.leaf_loss_quantify:
                Nkmin = sum(y_target == self.cl_no_red)
                root_source_values = get_node_distribution(self.T[i], 0).reshape(-1)

                props_s = root_source_values
                props_s = props_s / sum(props_s)
                props_t = np.zeros(props_s.size)
                for k in range(props_s.size):
                    props_t[k] = np.sum(y_target == k) / y_target.size

                coeffs = np.divide(props_t, props_s)

            # source_values_tot = rf_ser.estimators_[i].tree_.value[0,0,cl_no_red]

            inds = np.linspace(0, y_target.size - 1, y_target.size).astype(int)

            SER(0, self.T[i], X_target[inds], y_target[inds], original_ser=self.original_ser,
                no_red_on_cl=self.no_red_on_cl, cl_no_red=self.cl_no_red,
                no_ext_on_cl=self.no_ext_on_cl, cl_no_ext=self.cl_no_ext,
                ext_cond=self.ext_cond, leaf_loss_quantify=self.leaf_loss_quantify, leaf_loss_threshold=self.leaf_loss_threshold,
                coeffs=coeffs, root_source_values=root_source_values, Nkmin=Nkmin)

        # Recombine old and updated trees
        self.updated_T = [self.T[idx] for idx in T_old_indices] + [self.T[idx] for idx in T_new_indices]

    def weighted_majority_vote(self, x):
        votes = {}
        for i, tree in enumerate(self.updated_T):
            prob = tree.predict_proba(x)[0]
            for cls, p in enumerate(prob):
                if cls not in votes:
                    votes[cls] = 0
                if i >= len(self.updated_T) - self.n_update:  # These are the new trees
                    votes[cls] += self.w_new * p
                else:  # These are the old trees
                    votes[cls] += self.w_old * p
        return max(votes, key=votes.get)

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.weighted_majority_vote(x.reshape(1, -1)))
        return np.array(predictions)

    def score(self, X_test, y_test):

        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy