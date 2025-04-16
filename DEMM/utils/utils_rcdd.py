import numpy as np
import torch
import torch_sparse
# from utils import load_data, set_params, clustering_metrics,RSC
# from module.BMGC import *
from utils.preprocess import *
import warnings
import datetime
import random
from sklearn.cluster import KMeans
from scipy import linalg, sparse
from sklearn.utils import check_random_state
from sklearn.utils.extmath import  safe_sparse_dot,svd_flip
from utils.random_feature import OrthogonalRandomFeature as ORF
from utils.ADE import ADE
import  networkx as nx
from scipy.linalg import clarkson_woodruff_transform as cwt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import time
from scipy import io
import os
EPS=1e-10
def clustering_accuracy(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment

    def ordered_confusion_matrix(y_true, y_pred):
      conf_mat = confusion_matrix(y_true, y_pred)
      w = np.max(conf_mat) - conf_mat
      row_ind, col_ind = linear_sum_assignment(w)
      conf_mat = conf_mat[row_ind, :]
      conf_mat = conf_mat[:, col_ind]
      return conf_mat

    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    return np.trace(conf_mat) / np.sum(conf_mat)



## random seed ##
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def format_time(time):
    elapsed_rounded = int(round((time)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def process_feature(X,args):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    device = X.device
    X = X.cpu().numpy()

    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=args.dim,random_state=6)
    X = pca.fit_transform(X)
    # X = square_feat_map(X, 0.00001)
    X = torch.from_numpy(X).to(torch.float32)

    return X
def F_norm(A):
    if A is sparse:
        values=A._values()
        sum=torch.sum(values**2)
    else:
        sum=torch.sum(A.sum(1)**2)
    F=torch.sqrt(sum)

    # F=torch.norm(A,p='fro')
    return F
def l_norm(A,n):
    norms = torch.norm(A, p=n, dim=1, keepdim=True)
    return A/norms
def norm_w(w_lsit):
    sum=torch.norm(w_lsit,p=1)
    w_norm=w_lsit/sum
    return w_norm
def sinkhorn_knopp_adjust(Z):
    Zl = Z.clone()

    for _ in range(2):
        c = torch.matmul(Zl, torch.sum(Z.T, dim=1))
        c = 1.0 / c

        indices = torch.arange(len(c), device=c.device).unsqueeze(0).repeat(2, 1)
        values = c.flatten()
        diag_sparse = torch.sparse.FloatTensor(indices, values, torch.Size([len(c), len(c)]))
        Zl = torch.sparse.mm(diag_sparse, Zl)
        c = torch.matmul(torch.sum(Zl, dim=0), Z.T)
        c=1.0/c
        values = c.flatten()
        diag_sparse = torch.sparse.FloatTensor(indices, values, torch.Size([len(c), len(c)]))
        Z = torch.sparse.mm(diag_sparse, Z)

    return Z
def SSKC(H,args):
    # H=H*np.sqrt(mean_degree.cpu().numpy())[:, np.newaxis]
    if args.gamma!=0:
        orf = ORF(n_components=H.shape[1], gamma=args.gamma, distribution='gaussian', random_fourier=True,use_offset=False, random_state=42)
    else:
        orf = ORF(n_components=H.shape[1], gamma='auto', distribution='gaussian', random_fourier=True,
                  use_offset=False, random_state=42)
    orf.fit(H,seed=args.seed)
    H = orf.transform(H)

    H=sinkhorn_knopp_adjust(H)

    return H
def FFAO(adj,feature,L,alpha,sigma):
    para1=1/(alpha+1)
    para2=alpha/(alpha+1)
    feature0=feature
    for i in range(L):
        feature=para2*safe_sparse_dot(adj,feature)+feature0
        if i==(L-2):
            minus_A=feature

    remainder=feature-minus_A
    H=para2*remainder+para1*feature
    return H


def sum_set(adjs):
    sum=adjs[0]
    for i in range(1,len(adjs)-1):
        sum+=adjs[i]
    return sum
def sum_set(adjs):
    sum=adjs[0]
    for i in range(1,len(adjs)-1):
        sum+=adjs[i]
    return sum



def pcc_norm(A):
    mean_row=A.mean( dim=1, keepdim=True)
    A_norm = A - mean_row
    A_l2=torch.norm(A, p=2, dim=1, keepdim=True)
    A=A_norm/A_l2
    A=torch.nan_to_num(A)
    return A









def get_I(n,device):

    src=torch.arange(n)
    dst=torch.arange(n)
    indices=torch.stack((src,dst))
    values=torch.ones(n)
    adj_I=torch.sparse_coo_tensor(indices,values,(n,n)).to(device)
    return adj_I


def compu_H(adjs,H,args,n_class,incis,label,select_nodes):
    num_view = len(adjs)
    device = H.device
    omega = torch.full((len(adjs),), 1.0 / num_view, device=adjs[0].device)
    # omega=torch.tensor([0.5447,0.4553])
    # adj_I=get_I(adjs[0].size(0),device)
    #
    # adjs_I_minus=[(adj_I-adj).coalesce() for adj in adjs]
    X=H.clone().detach()
    empty_indices = torch.empty((2, 0), dtype=torch.long)
    empty_values = torch.empty((0,), dtype=torch.float32)
    shape = (H.shape[0], H.shape[0])
    AF_norm=[]
    for i in range(num_view):
        Af=torch.sum(adjs[i]._values()**2)
        AF_norm.append(Af)
    for i in range(args.n_time):
        # H=col_norm(H,2)
        # if i==0:
        #     H=l_norm(H,2)
        A = torch.sparse_coo_tensor(empty_indices, empty_values, shape, device=device)
        for j in range(num_view):
            weighted_adj = adjs[j] * (omega[j])
            A = A + weighted_adj
        A = A.coalesce()

        if i==args.n_time-1:
            print(omega)


        norm_A = normalize_adj_from_tensor(A, 'sym', True)
        H=FFAO(norm_A,X,args.L,args.alpha,args.sigma)
        H=l_norm(H,2)
        for j in range(num_view):
            omega[j]=compu_omega(H,adjs,AF_norm[j],args,incis[j])
        omega=norm_w(omega)

    return H

def compu_omega(H,adj,Af,args,incis):

    # H1=safe_sparse_dot(adj,H)
    # trace=0
    #
    # for i in range(H.shape[1]):
    #     trace+=torch.sum(H.T[i,:]*H1[:,i],dim=0)
        # print(torch.sum(H.T[i,:]*H1[:,i],dim=0))
        # if torch.sum(H.T[i,:]*H1[:,i],dim=0)<0:
        #     print(torch.sum(H.T[i,:]*H1[:,i],dim=0))
    H_m = torch.matmul(H.T, incis)
    # print(Af,F_norm(H_m)**2)
    trace = F_norm(H_m) ** 2
    if args.method=="demm+":
        c=(Af)*args.beta+args.sigma*trace
    else:
        c = torch.sqrt(Af) * args.beta + args.sigma * trace
    print(c,torch.sqrt(Af)*args.beta,args.sigma * trace)

    omega=c**(-2)

    return omega
def coo_tensor_to_csr(sparse_tensor: torch.Tensor) -> csr_matrix:

    if not sparse_tensor.is_sparse:
        raise ValueError("error")
    sparse_tensor = sparse_tensor.coalesce().cpu()

    rows = sparse_tensor.indices()[0].numpy()
    cols = sparse_tensor.indices()[1].numpy()
    data = sparse_tensor.values().numpy()

    sorted_order = rows.argsort()
    rows_sorted = rows[sorted_order]
    cols_sorted = cols[sorted_order]
    data_sorted = data[sorted_order]

    return csr_matrix(
        (data_sorted, (rows_sorted, cols_sorted)),
        shape=sparse_tensor.shape
    )
def compu_D(adj_csr):
    import scipy.sparse as sp
    degrees = adj_csr.sum(axis=1).A.ravel()
    # print(degrees)
    D = sp.diags(degrees, format="csr")
    degrees=1.0/np.sqrt(degrees)+EPS
    D_aqrt = sp.diags(degrees, format="csr")
    return D_aqrt,D
def csr_to_spcoot(csr,device):

    coo = csr.tocoo()

    rows = torch.tensor(coo.row, dtype=torch.long)
    cols = torch.tensor(coo.col, dtype=torch.long)
    indices = torch.stack([rows, cols], dim=0)
    data = torch.tensor(coo.data, dtype=torch.float32)

    return torch.sparse_coo_tensor(indices, data, csr.shape)
def get_inci(adjs,args,device):
    from scipy.sparse import csc_matrix
    num_view=len(adjs)
    incidences=[]
    for i in range(num_view):
        adj=coo_tensor_to_csr(adjs[i])
        # print(adj.nnz)
        D_inv_sqrt,D=compu_D(adj)
        G=nx.from_scipy_sparse_array(adj)
        # print(G.edges())
        # start_inc=time.time()
        inc = nx.incidence_matrix(G,oriented=True)# get E with (n*M)
        # end_inc=time.time()-start_inc
        # print('Get incidence matrix time: ',end_inc)
        # start_cwt=time.time()
        inci=cwt(inc.T,args.m[i],seed=42)# get E'
        # print('The num of edges in E:{}, The num of edges in E\':{}'.format(inc.nnz,inci.nnz))
        inci=D_inv_sqrt.dot(inci.T)
        # end_cwt=time.time()-start_cwt
        inci=csr_to_spcoot(inci,device)

        incidences.append(inci)

    return incidences
def KSI(L,dim,T=50):
    device=L.device
    n=L.size(0)
    torch.manual_seed(42)
    Z = torch.randn(size=(n, dim))
    Y,_=torch.linalg.qr(Z.to(device),mode='reduced')
    for i in range(T):
        Z=safe_sparse_dot(L,Y)
        Z=safe_sparse_dot(L.T,Z)
        Y,_=torch.linalg.qr(Z,mode='reduced')
    return Y

def top_k_eigh(adj,k,device):

    adj=adj.coalesce()
    
    eigenvectors=KSI(adj,k)
    return eigenvectors
def compu_H_al(adjs,H,args,incis,n_class):
    from sklearn.utils.extmath import randomized_svd
    num_view = len(adjs)
    device = H.device
    omega = torch.full((len(adjs),), 1.0 / num_view, device=adjs[0].device)
    # omega=torch.tensor([0.5447,0.4553])
    adj_I=get_I(adjs[0].size(0),device)

    adjs_I_minus=[(adj_I-adj).coalesce() for adj in adjs]
    X=H.clone().detach()
    empty_indices = torch.empty((2, 0), dtype=torch.long)
    empty_values = torch.empty((0,), dtype=torch.float32)
    shape = (H.shape[0], H.shape[0])
    AF_norm=[]
    for i in range(num_view):
        Af=torch.sum(adjs[i]._values()**2)
        AF_norm.append(Af)
    for i in range(args.n_time):
        
        A = torch.sparse_coo_tensor(empty_indices, empty_values, shape, device=device)
        for j in range(num_view):
            weighted_adj = adjs[j] * (omega[j])
            A = A + weighted_adj
        A = A.coalesce()
      

        norm_A = normalize_adj_from_tensor(A, 'sym', True)
        
        H=top_k_eigh(norm_A,args.dim,device)

        
        H=torch.tensor(H).to(device).float()

        
        if i==args.n_time-1:
            print(omega)
        for j in range(num_view):
            omega[j]=compu_omega(H,adjs_I_minus[j],AF_norm[j],args,incis[j])
        omega=norm_w(omega)


    return H
