import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn import metrics
from utils.munkres import Munkres, print_matrix

def cluster_acc(y_true, y_pred):

    y_true = np.array(y_true).astype('int')
    y_pred = y_pred.astype('int')
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D)) #, dtype=np.int64
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

class clustering_metrics():
    def __init__(self, true_label, predict_label, dataset):
        self.true_label = true_label
        self.pred_label = predict_label
        self.dataset = dataset


    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        return acc, nmi, f1_macro, adjscore


import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_array
from sklearn.utils.validation import check_consistent_length


def weighted_normalized_mutual_info_score(labels_true, labels_pred, weights=None, average_method="arithmetic"):
    """Weighted Normalized Mutual Information between two clusterings."""
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    if weights is None:
        weights = np.ones(len(labels_true))
    else:
        weights = check_array(weights, ensure_2d=False)
        check_consistent_length(labels_true, weights)

    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    # Special limit cases: no clustering since the data is not split.
    if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0):
        return 1.0

    contingency = weighted_contingency_matrix(labels_true, labels_pred, weights)

    # Calculate the weighted MI for the two clusterings
    mi = weighted_mutual_info_score(labels_true, labels_pred, weights, contingency=contingency)

    if mi == 0:
        return 0.0

    # Calculate weighted entropy for each labeling
    h_true, h_pred = weighted_entropy(labels_true, weights), weighted_entropy(labels_pred, weights)

    normalizer = _generalized_average(h_true, h_pred, average_method)
    return mi / normalizer


def weighted_contingency_matrix(labels_true, labels_pred, weights):
    """Build a weighted contingency matrix."""
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]

    contingency = sp.coo_matrix(
        (weights, (class_idx, cluster_idx)),
        shape=(n_classes, n_clusters),
        dtype=np.float64,
    )
    contingency = contingency.tocsr()
    contingency.sum_duplicates()
    return contingency


def weighted_mutual_info_score(labels_true, labels_pred, weights, contingency=None):
    if contingency is None:
        contingency = weighted_contingency_matrix(labels_true, labels_pred, weights)

    nzx, nzy = contingency.nonzero()
    nz_val = contingency.data

    total_weight = weights.sum()
    pi = np.ravel(contingency.sum(axis=1)) / total_weight
    pj = np.ravel(contingency.sum(axis=0)) / total_weight

    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / total_weight

    outer = pi.take(nzx) * pj.take(nzy)
    log_outer = np.log(outer)

    mi = np.sum(contingency_nm * (log_contingency_nm - np.log(total_weight) - log_outer))

    return mi


def weighted_entropy(labels, weights):
    """Calculates the weighted entropy of a labeling."""
    labels = np.asarray(labels)
    weights = np.asarray(weights)

    pi = np.bincount(labels, weights=weights)
    pi = pi[pi > 0]
    total_weight = weights.sum()

    # Calculate the weighted entropy
    return -np.sum((pi / total_weight) * np.log(pi / total_weight))


def _generalized_average(a, b, method):
    if method == "min":
        return min(a, b)
    elif method == "max":
        return max(a, b)
    elif method == "geometric":
        return np.sqrt(a * b)
    elif method == "arithmetic":
        return np.mean([a, b])
    else:
        raise ValueError("'method' must be 'min', 'max', 'arithmetic', or 'geometric'")


def check_clusterings(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0])
        )
    return labels_true, labels_pred
