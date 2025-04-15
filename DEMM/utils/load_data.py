import numpy as np
import scipy.sparse as sp
import torch

from sklearn.preprocessing import OneHotEncoder
import scipy.io as sio
from utils.preprocess import *
import pickle as pkl
import torch.nn.functional as F
import torch as th
def remap_label(label):
    unique_values=torch.unique(label)
    mapping = {value.item(): idx  for idx, value in enumerate(unique_values)}
    mapped_label = torch.tensor([mapping.get(x.item(), 0) for x in label])
    return mapped_label
def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def adj_values_one(adj):
    adj = adj.coalesce()
    index = adj.indices()
    return th.sparse.FloatTensor(index, th.ones(len(index[0])), adj.shape)

def sparse_tensor_add_self_loop(adj):
    adj = adj.coalesce()
    node_num = adj.shape[0]
    index = torch.stack((torch.tensor(range(node_num)), torch.tensor(range(node_num))), dim=0).to(adj.device)
    values = torch.ones(node_num).to(adj.device)

    adj_new = torch.sparse.FloatTensor(torch.cat((index, adj.indices()), dim=1), torch.cat((values, adj.values()),dim=0), adj.shape)
    return adj_new


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def sp_tensor_to_sp_csr(adj):
    adj = adj.coalesce()
    row = adj.indices()[0]
    col = adj.indices()[1]
    data = adj.values()
    shape = adj.size()
    adj = sp.csr_matrix((data, (row, col)), shape=shape)
    return adj



def load_acm_4019():

    path = "./data/acm-4019/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_p = sp.load_npz(path + "p_feat.npz")

    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    adjs = [pap, psp]
    adjs = [sparse_mx_to_torch_sparse_tensor(adj).coalesce() for adj in adjs]

    label = th.FloatTensor(label)

    feat_p = th.FloatTensor(preprocess_features(feat_p))

    return feat_p, adjs, label


def load_acm_3025():
    """load dataset ACM

    Returns:
        gnd(ndarray): [nnodes,]
    """

    # Load data
    dataset = "./data/acm-3025/" + 'ACM3025'
    data = sio.loadmat('{}.mat'.format(dataset))
    X = data['feature']
    A = data['PAP']
    B = data['PLP']

    if sp.issparse(X):
        X = X.todense()

    A = np.array(A)
    B = np.array(B)
    X = np.array(X)


    Adj = []
    Adj.append(A)
    Adj.append(B)

    gnd = data['label']
    gnd = gnd.T
    gnd = np.argmax(gnd, axis=0)

    X = torch.tensor(X).float()
    Adj = [torch.tensor(adj).to_sparse() for adj in Adj]
    # Adj = remove_self_loop(Adj)
    label = encode_onehot(gnd)
    label = th.FloatTensor(label)

    X = F.normalize(X, dim=1, p=2)

    return X, Adj, label


def load_dblp():
    path = "./data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")

    apa = sp.load_npz(path + "apa.npz")  
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    adjs = [apa, apcpa, aptpa]
    adjs = [sparse_mx_to_torch_sparse_tensor(adj).coalesce() for adj in adjs]
    
    label = th.FloatTensor(label)
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    rowsum = feat_a.sum(dim=1)
    r_inv = th.pow(rowsum, -1).flatten()
    r_inv[th.isinf(r_inv)] = 0.
    r_inv = r_inv.view(-1, 1)
    feat_a = feat_a * r_inv
    return feat_a, adjs, label


def load_amazon():
    data = pkl.load(open("data/amazon/amazon.pkl", "rb"))
    label = data['label'].argmax(1)
    label = encode_onehot(label)
    label = th.FloatTensor(label)

    # dense
    ivi = torch.from_numpy(data["IVI"]).float()
    ibi = torch.from_numpy(data["IBI"]).float()
    ioi = torch.from_numpy(data["IOI"]).float()
    adj = []
    adj.append(ivi)
    adj.append(ibi)
    adj.append(ioi)
    adj = [a.to_sparse().coalesce() for a in adj]

    X = torch.from_numpy(data['feature']).float()
    rowsum = X.sum(dim=1)
    r_inv = th.pow(rowsum, -1).flatten()
    r_inv[th.isinf(r_inv)] = 0.
    r_inv = r_inv.view(-1,1)
    X = X * r_inv
    return X, adj, label


def load_yelp():

    path = "./data/yelp/"
    feat_b = sp.load_npz(path + "features_0.npz").astype("float32")
    feat_b = th.FloatTensor(preprocess_features(feat_b))
    label = np.load(path+'labels.npy')
    label = encode_onehot(label)
    label = th.FloatTensor(label)

    blb = np.load(path+'blb.npy').astype("float32")
    bsb = np.load(path+'bsb.npy').astype("float32")
    bub = np.load(path+'bub.npy').astype("float32")


    adjs = [bsb, bub, blb]
    adjs = [th.tensor(adj).to_sparse() for adj in adjs]


    return feat_b, adjs, label



# def load_csbm_20():
#     r = 20
#
#     path = "./data/csbm/"
#     label = th.load(path+'label.pt')
#     label = F.one_hot(label, num_classes=2)
#
#     feat = th.load(path+'feat.pt').float()
#     adj_1 = th.load(path+'adj_v_0.pt').coalesce()
#     adj_2 = th.load(path+str(r)+'/adj_v_1.pt').coalesce()
#     adj_3 = th.load(path+str(r)+'/adj_v_2.pt').coalesce()
#     adjs = [adj_1, adj_2, adj_3]
#
#     return feat, adjs, label
#
#
#
# def load_csbm_50():
#     r = 50
#
#     path = "./data/csbm/"
#     label = th.load(path+'label.pt')
#     label = F.one_hot(label, num_classes=2)
#
#     feat = th.load(path+'feat.pt').float()
#     adj_1 = th.load(path+'adj_v_0.pt').coalesce()
#     adj_2 = th.load(path+str(r)+'/adj_v_1.pt').coalesce()
#     adj_3 = th.load(path+str(r)+'/adj_v_2.pt').coalesce()
#     adjs = [adj_1, adj_2, adj_3]
#
#     return feat, adjs, label
#
#
# def load_csbm_100():
#     r = 100
#
#     path = "./data/csbm/"
#     label = th.load(path+'label.pt')
#     label = F.one_hot(label, num_classes=2)
#
#     feat = th.load(path+'feat.pt').float()
#     adj_1 = th.load(path+'adj_v_0.pt').coalesce()
#     adj_2 = th.load(path+str(r)+'/adj_v_1.pt').coalesce()
#     adj_3 = th.load(path+str(r)+'/adj_v_2.pt').coalesce()
#     adjs = [adj_1, adj_2, adj_3]
#
#     return feat, adjs, label
#
#
#
# def load_csbm_150():
#     r = 150
#
#     path = "./data/csbm/"
#     label = th.load(path+'label.pt')
#     label = F.one_hot(label, num_classes=2)
#
#     feat = th.load(path+'feat.pt').float()
#     adj_1 = th.load(path+'adj_v_0.pt').coalesce()
#     adj_2 = th.load(path+str(r)+'/adj_v_1.pt').coalesce()
#     adj_3 = th.load(path+str(r)+'/adj_v_2.pt').coalesce()
#     adjs = [adj_1, adj_2, adj_3]
#
#     return feat, adjs, label

def load_mag():

    path = "./data/mag-4/"
    label = th.load(path+'label.pt')
    label = F.one_hot(label, num_classes=4)

    feat = th.load(path+'feat.pt').float()
    adj_1 = th.load(path+'pap.pt').coalesce()
    adj_2 = th.load(path+'pp.pt').coalesce()

    adjs = [adj_1, adj_2]

    return feat, adjs, label
def num_edge(adj,select_nodes):
    indice=adj._indices()
    mask=torch.isin(indice[0],select_nodes)
    str=indice[0][mask]
    num_nodes=len(torch.unique(str))
    return mask.sum(),num_nodes
def load_rcdd():
    from collections import Counter
    path = "./data/rcdd/"
    label = th.load(path+'label.pt')
    select_nodes = th.load(path + 'select_nodes.pt')
    feat = th.load(path+'feat.pt').float()
    adj_1 = th.load(path+'ibi.pt').coalesce()
    adj_2 = th.load(path+'ifi.pt').coalesce()
    # num1,num_nodes1=num_edge(adj_1,select_nodes)
    # num2,num_nodes2=num_edge(adj_2,select_nodes)
    # print(num1,num2)
    # print(num_nodes1,num_nodes2)
    # exit()
    adjs = [adj_1, adj_2]
    # feat[~select_nodes]=0
    label=label[select_nodes]
    # label = F.one_hot(label, num_classes=2)

    return feat, adjs, label,select_nodes
def load_cs():
    from collections import Counter
    path = "./data/oag-cs/"
    label = th.load(path+'label.pt')

    top_20_indice=th.load(path+'select_nodes.pt')
    feat = th.load(path+'feat.pt').float()
    adj_1 = th.load(path+'pp.pt').coalesce()
    adj_2 = th.load(path+'pap.pt').coalesce()
    adj_3=th.load(path+'pfp.pt').coalesce()
    adjs = [adj_1, adj_2,adj_3]

    # label=remap_label(label)

    label = F.one_hot(label, num_classes=20)
    return feat, adjs, label,top_20_indice
def load_eng():
    from collections import Counter
    path =  "./data/oag-eng/"
    label = th.load(path+'label.pt')

    top_20_indice=th.load(path+'select_nodes.pt')
    feat = th.load(path+'feat.pt').float()
    adj_1 = th.load(path+'pp.pt').coalesce()
    adj_2 = th.load(path+'pap.pt').coalesce()
    adj_3=th.load(path+'pfp.pt').coalesce()
    adjs = [adj_1, adj_2,adj_3]


    label = label[top_20_indice]
    label=remap_label(label)

    label = F.one_hot(label, num_classes=20)

    return feat, adjs, label,top_20_indice
def load_chem():
    from collections import Counter
    path = "/tmp/load_data/oag-chem/"
    label = th.load(path+'label.pt')

    top_20_indice=th.load(path+'select_nodes.pt')
    feat = th.load(path+'feat.pt').float()
    adj_1 = th.load(path+'pp.pt').coalesce()
    adj_2 = th.load(path+'pap.pt').coalesce()
    adj_3=th.load(path+'pfp.pt').coalesce()
    adjs = [adj_1, adj_2,adj_3]

    # label=remap_label(label)

    label = F.one_hot(label, num_classes=20)
    return feat, adjs, label,top_20_indice
def load_imdb():
    data = pkl.load(open("./data/imdb.pkl", "rb"))
    label = data['label']
###########################################################
    adj_edge1 = data["MDM"]
    adj_edge2 = data["MAM"]
    adj_fusion1 = adj_edge1 + adj_edge2
    # for i in range(0,3550):
    #     for j in range(0, 3550):
    #         if adj_fusion[i][j]!=2:
    #             adj_fusion[i][j]=0
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 2] = 0
    adj_fusion[adj_fusion == 2] = 1
    ############################################################
    adj1 = data["MDM"] + np.eye(data["MDM"].shape[0])
    adj2 = data["MAM"] + np.eye(data["MAM"].shape[0])
    adj_fusion = adj_fusion  + np.eye(adj_fusion.shape[0])*3
    # torch.where(torch.eq(torch.Tensor(adj1), 1) == True)
    adj1 = torch.from_numpy(adj1)
    adj2 = torch.from_numpy(adj2)
    adj_fusion = sp.csr_matrix(adj_fusion)

    # adj1_dense = torch.dense(adj1)
    adj_list=[adj1,adj2]
    truefeatures = data['feature']
    truefeatures = sp.lil_matrix(truefeatures)
    truefeatures = torch.Tensor(preprocess_features(truefeatures))

    # label = encode_onehot(label)
    label = th.FloatTensor(label)
    adj = [a.to_sparse().coalesce().to(torch.float) for a in adj_list]


    return  truefeatures, adj, label

def load_aminer():
    """load aminer

    Args:
        ratio (list, optional): _description_. Defaults to [20, 40, 60].
        type_num (list, optional): _description_. Defaults to [6564, 13329, 35890].

    Returns:
        label(ndarray): [nnodes, ]
    """
    path = "./data/aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    pos = sp.load_npz(path + "pos.npz")

    adj_pap = pap.todense().astype(int)
    adj_prp = prp.todense().astype(int)
    adj_pos = pos.todense().astype(int)
    adj = []
    adj.append(torch.from_numpy(adj_pap))
    adj.append(torch.from_numpy(adj_prp))

    feat_p = sp.eye(6564)
    feat_a = sp.eye(13329)
    feat_r = sp.eye(35890)
    feat_p = torch.Tensor(preprocess_features(feat_p))
    feat_a = torch.FloatTensor(preprocess_features(feat_a))
    feat_r = torch.FloatTensor(preprocess_features(feat_r))
    label = encode_onehot(label)
    label=th.FloatTensor(label)
    adj = [a.to_sparse().coalesce() for a in adj]
    return feat_p, adj, label

def load_data(dataset):
    if dataset == "acm-3025":
        data = load_acm_3025()
    elif dataset == "acm-4019":
        data = load_acm_4019()
    elif dataset == "dblp":
        data = load_dblp()
    elif dataset == 'amazon':
        data = load_amazon()
    elif dataset == 'yelp':
        data = load_yelp()
    # elif dataset == 'csbm-20':
    #     data = load_csbm_20()
    # elif dataset == 'csbm-50':
    #     data = load_csbm_50()
    # elif dataset == 'csbm-100':
    #     data = load_csbm_100()
    # elif dataset == 'csbm-150':
    #     data = load_csbm_150()
    elif dataset == 'mag':
        data = load_mag()
    elif dataset == 'aminer':
        data=load_aminer()
    elif dataset=='imdb':
        data=load_imdb()
    elif dataset == 'oag-cs':
        data=load_cs()
    elif dataset == 'oag-eng':
        data=load_eng()
    elif dataset == 'oag-chem':
        data=load_chem()
    elif dataset == 'rcdd':
        data=load_rcdd()
    return data



