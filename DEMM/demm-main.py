import torch
from utils import load_data, clustering_metrics

import warnings
import datetime
from sklearn.cluster import KMeans
import argparse
import time
from utils.utils_main import *
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from utils.preprocess import *
import os
import sys
import numpy as np
from scipy.sparse.linalg import eigsh

# os.environ['PYTHONHASHSEED'] = '0'
# os.execv(sys.executable, [sys.executable] + sys.argv)
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--load_parameters', default=True)
parser.add_argument('--dataset', type=str, default="acm-3025")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)


# model-specific parameters
parser.add_argument('--alpha', type=float, default=0.8)
# parser.add_argument('--sigma', type=float, default=0.8)
parser.add_argument('--n_T', type=int, default=10)
parser.add_argument('--n_time', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--dim', type=int, default=256)

parser.add_argument('--beta', type=float, default=0.1)

# parser.add_argument('--k', type=int, default=5)
# parser.add_argument(
#     "--k",
#     type=int,
#     nargs="+",
#     default=[5,5]
# )
args, _ = parser.parse_known_args()

if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

def matmul(H):
    n, d = H.shape
    output = torch.zeros((n, n), device=H.device, dtype=H.dtype)
    for i in range(n):
        for j in range(n):
            output[i, j] = torch.sum(H[i] * H[j])
    return output
def com_gaussian(H, sigma=1.0) :
    H=H.cpu()
    square_sum = torch.sum(H ** 2, dim = 1)


    dot_product = torch.matmul(H,H.T)
    # D=torch.matmul(matrix, matrix.T)


    distance_sq = square_sum.unsqueeze(1) + square_sum.unsqueeze(0) - 2 * dot_product
    SM = torch.exp(-distance_sq / sigma)
    inv_sqrt_degree = 1. / (torch.sqrt(SM.sum(dim=1, keepdim=False)) + EPS)
    SM=inv_sqrt_degree[:, None] * SM * inv_sqrt_degree[None, :]
    return SM
def com_k_eigenv(S,k):
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

    feat, adjs, label = load_data(args.dataset)
    nb_classes = label.shape[-1]
    args.n_classes = nb_classes
    # num_target_node = len(feat)
    #
    # feats_dim = feat.shape[1]
    # sub_num = int(len(adjs))
    # print("Dataset: ", args.dataset)
    # print("The number of meta-paths: ", sub_num)
    # print("Number of target nodes:", num_target_node)
    # print("The dim of target' nodes' feature: ", feats_dim)
    # print("Label: ", label.sum(dim=0))
    # print(args)

    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print(f'Using CUDA on {device}')
        adjs = [adj.to(device) for adj in adjs]
        feat = feat.to(device)


    # print(degree,mean_degree.shape)
    adjs_o,un_adjs = graph_process(adjs, feat, args)

    feat=process_feature(feat,args)


    start_time=time.time()


    start_inv = time.time()
    H=convergen_inv(adjs_o,feat,args,nb_classes)
    end_inv = time.time() - start_inv
    print("Time for compute H:{}".format(end_inv))


    label = torch.argmax(label, dim=-1)
    label = label.cpu().numpy()


    H = pcc_norm(H)

    start_S = time.time()
    S = com_gaussian(H, 1)

    S = com_k_eigenv(S,nb_classes)
    end_S=time.time() - start_S
    print("Time for compute S:{}".format(end_S))


    kmeans = KMeans(n_clusters=nb_classes,random_state=42)
    kmeans.fit(S)
    pre_labels = kmeans.labels_


    end_time = time.time() - start_time
    acc=clustering_accuracy(label, pre_labels)

    nmis=nmi(label, pre_labels)
    aris=ari(label, pre_labels)
    print("dataset:{},alpha:{},seed:{},gamma:{},dim:{},beta:{}".format(args.dataset,args.alpha,args.seed,args.gamma,args.dim,args.beta))
    print("ACC:{:.4},NMI:{:.4},ari:{:.4},time:{:.4}".format(acc,nmis,aris,end_time))
    output_file = '{}_results.txt'.format(args.dataset)

    with open(output_file, 'a') as f:
        f.write("dataset:{},alpha:{},seed:{},gamma:{},dim:{},beta:{}\n".format(args.dataset,args.alpha,args.seed,args.gamma,args.dim,args.beta))
        f.write("ACC:{:.4}, NMI:{:.4}, ARI:{:.4}, time:{:.4}\n".format(acc, nmis, aris, end_time))



if __name__:
    train()