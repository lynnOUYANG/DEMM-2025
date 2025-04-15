import numpy as np
import torch
from utils import load_data, set_params_large, clustering_metrics
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
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
from sklearn.cluster import KMeans
from utils.utils_large import *
import os
import psutil
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--load_parameters', default=True)
parser.add_argument('--dataset', type=str, default="mag")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
# parser.add_argument('--hidden_dim', type=int, default=256)
# parser.add_argument('--embed_dim', type=int, default=128)
# parser.add_argument('--nb_epochs', type=int, default=8)
# parser.add_argument('--nlayer', type=int, default=2)

# model-specific parameters
parser.add_argument('--alpha', type=float, default=0.8)
# parser.add_argument('--sigma', type=float, default=0.8)
# parser.add_argument('--n_T', type=int, default=10)
# parser.add_argument('--n_T2', type=int, default=10)
parser.add_argument('--n_time', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--beta', type=float, default=0.00001)
parser.add_argument('--max_h', type=int, default=1)
# parser.add_argument(
#     "--k",
#     type=int,
#     nargs="+",
#     default=[5,5,5]
# )
args, _ = parser.parse_known_args()

if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")
def com_aff(H, sigma=1.0) :
    H=H.cpu()
    square_sum = torch.sum(H ** 2, dim = 1)


    st_time=time.time()

    dot_product = torch.matmul(H,H.T)
    # D=torch.matmul(matrix, matrix.T)
    end_time=time.time()-st_time
    print("end time",end_time)

    distance_sq = square_sum.unsqueeze(1) + square_sum.unsqueeze(0) - 2 * dot_product
    SM = torch.exp(-distance_sq / sigma)
    inv_sqrt_degree = 1. / (torch.sqrt(SM.sum(dim=1, keepdim=False)) + EPS)
    SM=inv_sqrt_degree[:, None] * SM * inv_sqrt_degree[None, :]
    return SM
def com_k_eigenv(S,k):
    from scipy.sparse.linalg import eigsh

    S=S.numpy()
    np.random.seed(6)
    eigenvalues, eigenvectors = eigsh(S, k=k+1 , which='LM', v0=np.random.rand(S.shape[0]),maxiter=10e6,tol=0)
    sorted_indices = np.argsort(-eigenvalues)

    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    # print(sorted_eigenvalues)
    # print(eigenvalues,eigenvectors)
    # print(sorted_eigenvectors)
    return sorted_eigenvectors[:, 1:k + 1]
def train():
    if args.dataset in ["oag-cs","oag-eng"]:
        feat, adjs, label,select_nodes = load_data(args.dataset)
    else:
        feat, adjs, label = load_data(args.dataset)
    nb_classes = label.shape[-1]
    args.n_classes = nb_classes
    num_target_node = len(feat)

    feats_dim = feat.shape[1]
    sub_num = int(len(adjs))
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", sub_num)
    print("Number of target nodes:", num_target_node)
    print("The dim of target' nodes' feature: ", feats_dim)
    print("Label: ", label.sum(dim=0))
    print(args)


    adjs_o,un_adjs = graph_process_large(adjs, feat, args)

    feat = process_feature(feat, args)

    start_time = time.time()

    H = Brute_Force(adjs_o, feat, args, nb_classes)
    con_time=time.time() - start_time
    print("consensus time: ", con_time)
    label = torch.argmax(label, dim=-1)
    label = label.cpu().numpy()

    H = pcc_norm(H)
    # pcc_time=time.time()-start_time-con_time
    # print("pcc_time: ", pcc_time)

    S = com_aff(H, 1)

    Q = com_k_eigenv(S, nb_classes)
    if args.dataset in ["oag-cs","oag-eng"]:
        Q=Q[select_nodes]
    # Q = Q.numpy()
    # process_time = time.time() -con_time-pcc_time-start_time
    # print("process_time: ", process_time)


    kmeans = KMeans(n_clusters=nb_classes, random_state=42)
    kmeans.fit(Q)
    end_time = time.time() - start_time
    pre_labels = kmeans.labels_
    acc = clustering_accuracy(label, pre_labels)

    nmis = nmi(label, pre_labels)
    aris = ari(label, pre_labels)
    print("dataset:{},alpha:{},dim:{}\n".format(args.dataset, args.alpha, args.dim))
    print("ACC:{:.4},NMI:{:.4},ari:{:.4},time:{:.4}".format(acc, nmis, aris, end_time))
    output_file = '{}_results.txt'.format(args.dataset)

    with open(output_file, 'a') as f:
        f.write("dataset:{}, alpha:{}, dim:{}, beta:{} \n".format(args.dataset, args.alpha, args.hidden_dim,args.beta))
        f.write("ACC:{:.4}, NMI:{:.4}, ARI:{:.4}, time:{:.4}\n".format(acc, nmis, aris, end_time))



if __name__ == '__main__':
    train()


