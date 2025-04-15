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
import psutil

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

os.environ['PYTHONHASHSEED'] = '0'
# os.execv(sys.executable, [sys.executable] + sys.argv)
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
# parser.add_argument('--load_parameters', default=True)
parser.add_argument('--dataset', type=str, default="acm-3025")
parser.add_argument('--method', type=str, default="demm+")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
# parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--alpha', type=float, default=0.8)
# parser.add_argument('--sigma', type=float, default=0.8)
parser.add_argument('--L', type=int, default=10)
# parser.add_argument('--n_T2', type=int, default=10)
parser.add_argument('--n_time', type=int, default=10)
# parser.add_argument('--fusion', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--lamda', type=float, default=1.0)
parser.add_argument('--round', type=int, default=5)
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--max_h', type=int, default=3)
parser.add_argument('--decay', type=float, default=0.)
# parser.add_argument('--k', type=int, default=5)
parser.add_argument(
    "--m",
    type=int,
    nargs="+",
    default=[5,5,5]
)
args, _ = parser.parse_known_args()

if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")



def train():

    feat, adjs, label = load_data(args.dataset)
    nb_classes = label.shape[-1]
    args.n_classes = nb_classes
    feats_dim = feat.shape[1]
    num_target_node = len(feat)
    sub_num = int(len(adjs))
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", sub_num)
    print("Number of target nodes:", num_target_node)
    print("The dim of target' nodes' feature: ", feats_dim)
    print("Label: ", label.sum(dim=0))
    print(args)


    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print(f'Using CUDA on {device}')
        adjs = [adj.to(device) for adj in adjs]
        feat = feat.to(device)

    adjs_o,un_adjs = graph_process(adjs, feat, args)


    incis = get_inci(un_adjs, args, device)

    feat=process_feature(feat,args)
    start_time=time.time()

    if args.method=="demm+":
        H=compu_H(adjs_o,feat,args,n_class=nb_classes,incis=incis)
    else:
        H = compu_H_al(adjs_o,feat,args,incis,nb_classes)

    label = torch.argmax(label, dim=-1)
    label = label.cpu().numpy()
    # for alpha in alpha_list:
    #     for T in T_list:

    Q=H
    Q=Q.cpu()
    Q = pcc_norm(Q)
    Q = SSKC(Q,args)
    Q = Q.numpy()

    kmeans = KMeans(n_clusters=nb_classes,random_state=42)
    kmeans.fit(Q)
    end_time = time.time() - start_time
    pre_labels = kmeans.labels_
    acc=clustering_accuracy(label, pre_labels)

    nmis=nmi(label, pre_labels)
    aris=ari(label, pre_labels)
    print("dataset:{},alpha:{},L:{},seed:{},dim:{},beta:{},m:{}".format(args.dataset,args.alpha,args.L,args.seed,args.dim,args.beta,args.m))
    print("ACC:{:.4},NMI:{:.4},ari:{:.4},time:{:.4}".format(acc,nmis,aris,end_time))
    output_file = '{}_results.txt'.format(args.dataset)

    with open(output_file, 'a') as f:
        f.write("dataset:{},alpha:{},L:{},seed:{},dim:{},decay:{},beta:{}\n".format(args.dataset,args.alpha,args.L,args.seed,args.dim,args.decay,args.beta))
        f.write("ACC:{:.4}, NMI:{:.4}, ARI:{:.4}, time:{:.4}\n".format(acc, nmis, aris, end_time))



if __name__:
    train()


