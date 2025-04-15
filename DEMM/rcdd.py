import numpy as np
import torch
from utils import load_data, set_params_large, clustering_metrics

from utils.preprocess import *
import warnings
import datetime
import time
import random
from kmeans_pytorch import kmeans
from torch.utils.data import RandomSampler
import argparse
from scipy import linalg, sparse
from sklearn.utils.extmath import  safe_sparse_dot
from sklearn.cluster import KMeans,DBSCAN,MeanShift
from torch_geometric.loader import NeighborSampler
from utils.utils_rcdd import *
from utils.metrics import weighted_normalized_mutual_info_score as wnmi
import psutil
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from scipy import io
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
# parser.add_argument('--load_parameters', default=True)
parser.add_argument('--dataset', type=str, default="rcdd")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
# parser.add_argument('--hidden_dim', type=int, default=256)
# parser.add_argument('--embed_dim', type=int, default=128)
parser.add_argument('--nb_epochs', type=int, default=8)
# parser.add_argument('--nlayer', type=int, default=2)

# parser.add_argument('--l2_coef', type=float, default=1e-4)
# parser.add_argument('--lr', type=float, default=0.001)
# parser.add_argument('--dropout', type=float, default=0.)
# parser.add_argument('--tau', type=float, default=1.0)

# model-specific parameters
parser.add_argument('--alpha', type=float, default=0.8)
# parser.add_argument('--sigma', type=float, default=0.8)
parser.add_argument('--L', type=int, default=10)
parser.add_argument('--n_time', type=int, default=5)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--theta', type=int, default=20)
parser.add_argument('--round', type=int, default=5)
parser.add_argument('--beta', type=float, default=0.00001)
parser.add_argument('--method', type=str, default="demm+")

parser.add_argument(
    "--m",
    type=int,
    nargs="+",
    default=[5,5,5]
)
args, _ = parser.parse_known_args()


class WeightedKMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None

    def fit(self, X, sample_weights):
        n_samples, n_features = X.shape
        # Initialize cluster centers
        kmeans = KMeans(n_clusters=self.n_clusters, max_iter=1, random_state=42).fit(X)
        self.cluster_centers_ = kmeans.cluster_centers_

        for _ in range(self.max_iter):
            # Compute weighted distances
            distances = np.zeros((n_samples, self.n_clusters))
            for i in range(self.n_clusters):
                distances[:, i] = np.linalg.norm(X - self.cluster_centers_[i], axis=1) * sample_weights

            # Assign clusters
            labels = np.argmin(distances, axis=1)

            # Update cluster centers
            new_centers = np.zeros((self.n_clusters, n_features))
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                cluster_weights = sample_weights[labels == i]
                if len(cluster_points) > 0:
                    new_centers[i] = np.average(cluster_points, axis=0, weights=cluster_weights)

            # Check for convergence
            if np.linalg.norm(self.cluster_centers_ - new_centers) < self.tol:
                break
            self.cluster_centers_ = new_centers

    def predict(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - self.cluster_centers_[i], axis=1)
        return np.argmin(distances, axis=1)
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")
def train():

    feat, adjs, label ,select_nodes= load_data(args.dataset)

    nb_classes = len(torch.unique(label))

    num_target_node = len(feat)

    feats_dim = feat.shape[1]
    sub_num = int(len(adjs))
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", sub_num)
    print("Number of target nodes:", num_target_node)
    print("The dim of target' nodes' feature: ", feats_dim)
    print(args)

    # if torch.cuda.is_available():
    #     print(f'Using CUDA on {device}')
    #     adjs = [adj.to(device) for adj in adjs]
    #     feat = feat.to(device)
    start_process=time.time()
    adjs_o, un_adjs = graph_process_large(adjs, feat, args)
    incis = get_inci(un_adjs, args, device)

    # incis = torch.load('incis.pt')
    feat = process_feature(feat, args)
    end_process=time.time()-start_process
    print("Time taken to process: ", end_process)


    start_time = time.time()

    if args.method == "demm+":
        H = compu_H(adjs_o, feat, args, n_class=nb_classes, incis=incis,label=label,select_nodes=select_nodes)
    else:
        H = compu_H_al(adjs_o, feat, args, incis, nb_classes)
    consensus_time = time.time()-start_time
    print('Time taken to consensus: ',consensus_time)
    H = pcc_norm(H)
    H = SSKC(H, args)
    Q=H[select_nodes]
    label=label.numpy()
    class_counts = np.bincount(label)
    weights = 1.0 / class_counts[label]
    wkmeans = WeightedKMeans(n_clusters=nb_classes)
    wkmeans.fit(Q, weights)
    pre_labels =wkmeans.predict(Q)
    end_time = time.time() - start_time
    acc = clustering_accuracy(label, pre_labels)

    nmii = nmi(label, pre_labels)
    arii = ari(label, pre_labels)

    print("dataset:{},alpha:{},L:{},dim:{},beta:{}".format(args.dataset, args.alpha, args.L, args.dim,args.beta))
    print("ACC:{:.4},NMI:{:.4},ari:{:.4},time:{:.4}".format(acc, nmii, arii,end_time))
    output_file = '{}_results.txt'.format(args.dataset)

    with open(output_file, 'a') as f:
        f.write("dataset:{}, alpha:{},L:{}, dim:{},beta:{}\n".format(args.dataset, args.alpha, args.L, args.dim,args.beta))
        f.write("ACC:{:.4}, NMI:{:.4}, ARI:{:.4}, time:{:.4}\n".format(acc, nmii, arii, end_time))



if __name__ == '__main__':
    train()


