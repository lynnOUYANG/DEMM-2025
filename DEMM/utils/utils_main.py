import time

import numpy as np
import torch
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
from scipy.sparse import csr_matrix
import  networkx as nx
from scipy.linalg import clarkson_woodruff_transform as cwt
from scipy.sparse.linalg import eigsh
from sklearn.utils import check_random_state, as_float_array
from scipy.sparse import csc_matrix
from torch.utils.dlpack import from_dlpack

EPS=1e-15
def F_norm(A):
    if A is sparse:
        values = A._values()
        sum = torch.sum(values ** 2)
    else:
        sum = torch.sum(A**2)
    F = torch.sqrt(sum)

    # F=torch.norm(A,p='fro')
    return F


## random seed ##
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def format_time(time):
    elapsed_rounded = int(round((time)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def l_norm(A, n):
    norms = (torch.norm(A, p=n, dim=1, keepdim=True)+EPS)
    return A / norms
def col_norm(A, n):
    norms = (torch.norm(A, p=n, dim=0, keepdim=True)+EPS)
    return A / norms

def pcc_norm(A):
    mean_row = A.mean(dim=1, keepdim=True)
    A_norm = A - mean_row
    A_l2 = torch.norm(A, p=2, dim=1, keepdim=True)
    A = A_norm / A_l2
    A = torch.nan_to_num(A)
    return A




def filter_try(adj, feature, n_T, alpha):
    feature0 = feature.clone().detach()

    for _ in range(n_T):
        feature = alpha * (safe_sparse_dot(adj, feature)) + feature0

    return feature


def norm_w(w_list):
    sum=torch.norm(w_list,p=1)
    w_norm=w_list/sum
    return w_norm

def norm_w2(w_list):
    sum=torch.norm(w_list,p=2)
    w_norm=w_list/sum
    return w_norm



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




def sinkhorn_knopp_adjust(Z):
    Zl = Z.clone()

    for _ in range(2):
        c = torch.matmul(Zl, torch.sum(Z.T, dim=1))
        # c = 1.0 / torch.sqrt(c)
        c = 1.0 / c
        Zl = torch.matmul(torch.diag(c.flatten()), Zl)
        c = torch.matmul(torch.sum(Zl, dim=0), Z.T)
        c = 1.0 / c
        Z = torch.matmul(torch.diag(c.flatten()), Z)

    return Z

def SSKC(H, args):
    # H=H*np.sqrt(mean_degree.cpu().numpy())[:, np.newaxis]
    if args.gamma != 0:
        orf = ORF(n_components=H.shape[1], gamma=args.gamma, distribution='gaussian', random_fourier=True,
                  use_offset=False, random_state=42)
    else:
        orf = ORF(n_components=H.shape[1], gamma='auto', distribution='gaussian', random_fourier=True,
                  use_offset=False, random_state=42)

    orf.fit(H, seed=args.seed)

    H = orf.transform(H)
    st_time=time.time()
    H = sinkhorn_knopp_adjust(H)
    sk_time=time.time()-st_time
    print("sk time:",sk_time)
    # H=shift_matrix(H)
    # H=sinkhorn_knopp_adjust_simple(H)
    return H


def process_feature(X, args):
    from sklearn.decomposition import PCA,SparsePCA
    from sklearn.preprocessing import StandardScaler
    device = X.device
    X = X.cpu().numpy()

    # X = StandardScaler().fit_transform(X)
    #
    # pca = PCA(n_components=args.dim, random_state=6)
    # X = pca.fit_transform(X)
    # scaler = StandardScaler(with_mean=True, with_std=False)
    X_centered = StandardScaler().fit_transform(X)
    pca = PCA(n_components=args.dim, random_state=6)
    X = pca.fit_transform(X_centered)
    # if args.tau==0:
    #     spca = SparsePCA(n_components=4, alpha=1, random_state=6)
    #     X= spca.fit_transform(X_centered)

    # X = square_feat_map(X, 0.00001)
    X = torch.from_numpy(X).to(device).to(torch.float32)
    # print(X)
    return X


def FAAO(adj, feature, L, alpha):
    para1 = 1.0 / (1.0 + alpha)
    para2 = alpha / (1.0 + alpha)
    feature0 = feature
    for i in range(L):
        feature = para2 * safe_sparse_dot(adj, feature) + feature0
        if i == (L - 2):
            minus_A = feature

    remainder = feature - minus_A
    H = para2 * remainder + para1 * feature
    return H




def svd_flip(u, v, u_based_decision=True):
    if u_based_decision:
        # Determine signs based on u's columns (u.T's rows)
        max_abs = torch.abs(u.t())
        max_indices = torch.argmax(max_abs, dim=1)
        # Compute the sign of the maximum value in each column of u
        # Create row indices to pair with max_indices
        row_indices = torch.arange(max_abs.size(0), device=u.device)
        # Flatten u.T to a 1D tensor and get the values at the max indices
        signs = torch.sign(u.reshape(-1)[max_indices + row_indices * u.size(0)])
        # Apply the signs to u and v
        u = u * signs.view(1, -1)
        if v is not None:
            v = v * signs.view(-1, 1)
    else:
        # Determine signs based on v's rows
        max_abs = torch.abs(v)
        max_indices = torch.argmax(max_abs, dim=1)
        row_indices = torch.arange(max_abs.size(0), device=v.device)
        # Flatten v to a 1D tensor and get the values at the max indices
        signs = torch.sign(v.reshape(-1)[max_indices + row_indices * v.size(1)])
        # Apply the signs to u and v
        if u is not None:
            u = u * signs.view(1, -1)
        v = v * signs.view(-1, 1)

    return u, v


def get_I(n,device):

    src=torch.arange(n)
    dst=torch.arange(n)
    indices=torch.stack((src,dst))
    values=torch.ones(n)
    adj_I=torch.sparse_coo_tensor(indices,values,(n,n)).to(device)
    return adj_I




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


def csr_to_spcoot(csr,device):

    coo = csr.tocoo()

    rows = torch.tensor(coo.row, dtype=torch.long)
    cols = torch.tensor(coo.col, dtype=torch.long)
    indices = torch.stack([rows, cols], dim=0)
    data = torch.tensor(coo.data, dtype=torch.float32)

    return torch.sparse_coo_tensor(indices, data, csr.shape).to(device)


def compu_D(adj_csr):
    import scipy.sparse as sp
    degrees = adj_csr.sum(axis=1).A.ravel()
    # print(degrees)
    D = sp.diags(degrees, format="csr")
    degrees=1.0/np.sqrt(degrees)+EPS
    D_aqrt = sp.diags(degrees, format="csr")
    return D_aqrt,D

def get_inci(adjs,args,device):
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
def compu_omega(H,adj,Af,args,incis):

    # H1=safe_sparse_dot(adj,H)
    # trace=0
    #
    # for i in range(H.shape[1]):
    #     trace+=torch.sum(H.T[i,:]*H1[:,i],dim=0)
    # print(trace)
        # print(torch.sum(H.T[i,:]*H1[:,i],dim=0))
        # if torch.sum(H.T[i,:]*H1[:,i],dim=0)<0:
        #     print(torch.sum(H.T[i,:]*H1[:,i],dim=0))
    H_m=torch.matmul(H.T,incis)

    # print(Af,F_norm(H_m)**2)
    trace=F_norm(H_m)**2
    # for i in range(H_m.shape[0]):
    #     trace+=torch.sum(H_m[i,:]*H_m.T[:,i],dim=0)
    if args.method=="demm+":
        c=args.alpha*trace+Af*args.beta
    else:
        c = torch.sqrt(Af) * args.beta + args.alpha * trace
    print(args.alpha*trace,torch.sqrt(Af) * args.beta)
    # c = args.sigma * trace
    # omega=c**(-2)
    # trace2=0
    # print(H,incis)
    # H_m=torch.matmul(H.T,incis)
    # print(Af,F_norm(H_m)**2)
    # for i in range(H_m.shape[0]):
    #     trace2+=torch.sum(H_m[i,:]*H_m.T[:,i],dim=0)
    # c=Af*args.beta+args.sigma*(F_norm(H_m))**2
    # c=args.sigma*(F_norm(H_m)**2)
    # print(F_norm(H_m)**2)
    omega=c**(-2)

    return omega
def compu_omega_inv(H,adj,Af,args):

    H1=safe_sparse_dot(adj,H)
    trace=0

    for i in range(H.shape[1]):
        trace+=torch.sum(H.T[i,:]*H1[:,i],dim=0)

    c = args.alpha * trace + Af * args.beta
    print(args.alpha*trace,Af * args.beta)

    omega=c**(-2)

    return omega





def compu_H(adjs,H,args,n_class,incis):
    num_view = len(adjs)
    device = H.device
    omega = torch.full((len(adjs),), 1.0 / num_view, device=adjs[0].device)
    # omega=torch.tensor([0.5447,0.4553])
    # adj_I=get_I(adjs[0].size(0),device)
    #
    # adjs_I_minus=[(adj_I-adj).coalesce() for adj in adjs]
    X=H
    empty_indices = torch.empty((2, 0), dtype=torch.long)
    empty_values = torch.empty((0,), dtype=torch.float32)
    shape = (H.shape[0], H.shape[0])
    AF_norm=[]
    #get ||A_F||
    for i in range(num_view):
        Af=torch.sum(adjs[i]._values()**2)
        AF_norm.append(Af)

    for i in range(args.n_time):
        # H=col_norm(H,2)
        # if i==0:
        #     H=l_norm(H,2)
        A = torch.sparse_coo_tensor(empty_indices, empty_values, shape, device=device)
        #get \hat{A}
        for j in range(num_view):
            weighted_adj = adjs[j] * (omega[j])
            A = A + weighted_adj
            del weighted_adj
        A = A.coalesce()

        # A=0.5*adjs[0]+0.5*adjs[1]
        # U_c,U = sub_randomized_svd(H, n_components=args.hidden_dim)

        if i==args.n_time-1:
            print(omega)
        # A=reweight_adjacency_batched(A,H,args.p,1000000)
        # A,_=add_sim(args,U,A,1000000)

        norm_A = normalize_adj_from_tensor(A, 'sym', True)

        H=FAAO(norm_A,X,args.L,args.alpha)
        # del norm_A,A
        H=l_norm(H,2)

        for j in range(num_view):
            omega[j]=compu_omega(H,adjs,AF_norm[j],args,incis[j])

        omega=norm_w(omega)

    # A=(norm_A).to_dense()
    # L=torch.tensor([0])
    # mu=compute_max_with_corresponding_L(A,L)
    # print(mu)
    # exit()
        # H=filter_try(A,X,args.n_T,args.alpha)
    return H

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


def SNEM_rounding(vectors, t=100):
    def to_one_hot(y, n_classes=None):

        if n_classes is None:
            n_classes = np.max(y) + 1
        return np.eye(n_classes, dtype=int)[y]
    vectors = as_float_array(vectors)
    n_samples = vectors.shape[0]
    n_feats = vectors.shape[1]
    # print("n_samples:", n_samples)
    # print("n_feats:", n_feats)
    labels = vectors.argmax(axis=1)

    # print(type(labels), labels.shape)
    vectors_discrete = csc_matrix(
        (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
        shape=(n_samples, n_feats))

    vectors_sum = np.sqrt(vectors_discrete.sum(axis=0))
    vectors_sum[vectors_sum == 0] = 1
    vectors_discrete = vectors_discrete * 1.0 / vectors_sum
    # vectors_discrete = preprocessing.normalize(vectors_discrete, norm='l2', axis=0)
    # vectors_discrete = vectors_discrete.toarray()
    # print('compute the sqrt')
    for _ in range(t):

        Q = vectors.T.dot(vectors_discrete.toarray())
        # Q = np.matrix(Q)
        # Q = np.asarray(Q)
        # print("to array")
        # print(Q,vectors)
        # print(Q.shape,vectors.shape)

        vectors_discrete = vectors.dot(Q)
        # print("v dot Q")
        # vectors_discrete = as_float_array(vectors_discrete)

        labels = vectors_discrete.argmax(axis=1)
        vectors_discrete = csc_matrix(
            (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
            shape=(n_samples, n_feats))

        vectors_sum = np.sqrt(vectors_discrete.sum(axis=0))
        vectors_sum[vectors_sum == 0] = 1
        vectors_discrete = vectors_discrete * 1.0 / vectors_sum
    vectors_discrete = vectors_discrete.toarray()
    # one_hot=to_one_hot(labels)
    return labels
def sp_gap(L,args):
    eigenvalues = torch.linalg.eigvals(L)

    eigenvalues_real = eigenvalues.real
    sorted_eigenvalues = torch.sort(eigenvalues_real)[0]
    second_smallest_eigenvalue = sorted_eigenvalues[1]
    la_eig=sorted_eigenvalues[-200:]
    print("dataset:{},gap1:{},gap2:{}".format(args.dataset,second_smallest_eigenvalue,la_eig) )
def compu_H_al(adjs,H,args,incis,n_class):

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


def filter_inv(adj,X,alpha,adj_I):
    para1=1.0/(1.0+alpha)
    para2=alpha/(1.0+alpha)

    inv_adj=torch.inverse((adj_I-para2*adj).to_dense())
    H=para1*safe_sparse_dot(inv_adj,X)

    return H
def convergen_inv(adjs,H,args,n_class):
    num_view = len(adjs)
    device = H.device
    omega = torch.full((len(adjs),), 1.0 / num_view, device=adjs[0].device)

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


        if i==args.n_time-1:
            print(omega)


        norm_A = normalize_adj_from_tensor(A, 'sym', True)

        H=filter_inv(norm_A,X,args.alpha,adj_I)

        H=l_norm(H,2)

        for j in range(num_view):
            omega[j] = compu_omega_inv(H,adjs_I_minus[j],AF_norm[j],args)
        omega=norm_w(omega)

    return H
