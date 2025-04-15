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
import  networkx as nx
from scipy.linalg import clarkson_woodruff_transform as cwt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy import io
def sample_subgraph(adj_all, features, center_nodes, num_hops):
    device=adj_all.device
    current_nodes = torch.unique(center_nodes)
    all_nodes = current_nodes.clone()

    nodes_at_k_minus_1 = None

    for i in range(num_hops):
        mask = torch.isin(adj_all._indices()[0], current_nodes)
        sampled_edges = adj_all._indices()[:, mask]

        current_nodes = torch.unique(sampled_edges[1])

        all_nodes = torch.unique(torch.cat([all_nodes, current_nodes]))

        if i == (num_hops // 2 - 1):
            nodes_at_k_minus_1 = all_nodes.clone()

    mask = torch.isin(adj_all._indices()[0], all_nodes) & torch.isin(adj_all._indices()[1], all_nodes)
    subgraph_edges = adj_all._indices()[:, mask]
    subgraph_values = adj_all._values()[mask]
    # map_start=time.time()
    node_mapping_array = np.zeros(max(all_nodes) + 1, dtype=int)
    for idx, node in enumerate(all_nodes):
        node_mapping_array[node.item()] = idx
    node_mapping = torch.tensor(node_mapping_array, device=device)
    # node_mapping = {node.item(): idx for idx, node in enumerate(all_nodes)}
    src = subgraph_edges[0].to(device)
    src = node_mapping[src]

    dst = subgraph_edges[1].to(device)
    dst = node_mapping[dst]
    sizes=len(all_nodes)
    indices = torch.stack((src, dst), dim=0).to(device)
    # end_map=time.time()-map_start
    # print("Elapsed time:{}".format(end_map))
    subgraph = torch.sparse_coo_tensor(indices, subgraph_values, (sizes, sizes)).to(device)
    subgraph=subgraph+subgraph.transpose(0,1)
    inverse_node_mapping = torch.tensor(all_nodes, device=device)

    nodes_at_k_minus_1_mapped = node_mapping[nodes_at_k_minus_1]
    subgraph_features = features[all_nodes]

    return subgraph, subgraph_features, inverse_node_mapping, nodes_at_k_minus_1_mapped
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
    X = torch.from_numpy(X).to(device).to(torch.float32)

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

    for _ in range(10):
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
def FFAO(adj,feature,L,alpha):
    para1=1/(1+alpha)
    para2=alpha/(alpha+1)
    feature0=feature
    for i in range(L):
        feature=para2*safe_sparse_dot(adj,feature)+feature0
        if i==(L-2):
            minus_A=feature

    remainder=feature-minus_A
    H=para2*remainder+para1*feature
    # H = para1 * feature
    return H
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
# def compu_omega(H,adj,Af,args,incis):
def compu_omega(H, adj, Af, args,incis):
    # H1=safe_sparse_dot(adj,H)
    # trace=0
    #
    # for i in range(H.shape[1]):
    #     trace+=torch.sum(H.T[i,:]*H1[:,i],dim=0)
        # print(torch.sum(H.T[i,:]*H1[:,i],dim=0))
        # if torch.sum(H.T[i,:]*H1[:,i],dim=0)<0:
        #     print(torch.sum(H.T[i,:]*H1[:,i],dim=0))
    H_m=torch.matmul(H.T,incis)

    trace=F_norm(H_m)**2
    if args.method=="demm+":
        c=(Af)*args.beta+args.alpha*trace
    else:
        c = torch.sqrt(Af) * args.beta + args.alpha * trace
    print((Af)*args.beta,args.alpha*trace)
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

    return torch.sparse_coo_tensor(indices, data, csr.shape).to(device)
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
def compu_H(adjs,H,args,n_class,incis):
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
        # H=col_norm(H,2)
        if i==0:
            H=l_norm(H,2)
        A = torch.sparse_coo_tensor(empty_indices, empty_values, shape, device=device)
        for j in range(num_view):
            weighted_adj = adjs[j] * (omega[j])
            A = A + weighted_adj
        A = A.coalesce()



        if i==args.n_time-1:
            print(omega)


        norm_A = normalize_adj_from_tensor(A, 'sym', True)
        H=FFAO(norm_A,X,args.L,args.alpha)

        # label = (label + 1).numpy().astype(np.float64).reshape(-1, 1)
        #
        # select_nodes=select_nodes.numpy().astype(np.float64).reshape(-1, 1)
        # print(np.unique(select_nodes))
        # print(H1.shape[0])
        # exit()
        # H1 = H1.cpu().numpy().astype(np.float64)
        # H2 = H2.cpu().numpy().astype(np.float64)
        # H3 = H3.cpu().numpy().astype(np.float64)
        # matlab_cell = np.zeros((3, 1), dtype=object)
        # matlab_cell[0, 0] = H1
        # matlab_cell[1, 0] = H2
        # matlab_cell[2, 0] = H3
        # save_dict = {
        #     'data': matlab_cell,
        #     'label': label,
        #     'select_nodes':select_nodes
        # }
        #
        # io.savemat('{}.mat'.format(args.dataset), save_dict)
        # exit()
        H=l_norm(H,2)
        for j in range(num_view):
            omega[j]=compu_omega(H,adjs_I_minus[j],AF_norm[j],args,incis[j])
        omega=norm_w(omega)
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

def top_k_eigh(adj,k):

    adj=adj.coalesce()

    # csr_adj = csr_matrix(
    #     (adj.values().cpu().numpy(), (adj.indices()[0].cpu().numpy(), adj.indices()[1].cpu().numpy())),
    #     shape=adj.size()
    # )
    # np.random.seed(6)
    # eigenvalues, eigenvectors = eigsh(csr_adj, k=k+1 , which='LM', v0=np.random.rand(csr_adj.shape[0]))
    # # eigenvalues, eigenvectors = eigsh(csr_adj, k=k+1, which='LM')
    #
    # # eigenvectors_torch = from_dlpack(eigenvectors.toDlpack()).to(device)
    # # eigenvalues_torch = torch.tensor(eigenvalues, device=device)
    # # abs_eigenvalues = np.abs(eigenvalues)
    # sorted_indices = np.argsort(-eigenvalues)
    #
    # sorted_eigenvalues = eigenvalues[sorted_indices]
    # sorted_eigenvectors = eigenvectors[:, sorted_indices]
    # print(sorted_eigenvalues[1:k+1])
    # print(eigenvalues,eigenvectors)
    # print(sorted_eigenvectors)
    eigenvectors=KSI(adj,k)
    return eigenvectors
def compu_H_al(adjs,H,args,incis,n_class):

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
        # H=col_norm(H,2)
        if i==0:
            H=l_norm(H,2)
        A = torch.sparse_coo_tensor(empty_indices, empty_values, shape, device=device)
        for j in range(num_view):
            weighted_adj = adjs[j] * (omega[j])
            A = A + weighted_adj
        A = A.coalesce()
        # A=0.5*adjs[0]+0.5*adjs[1]
        # U_c,U = sub_randomized_svd(H, n_components=args.hidden_dim)


        # A=reweight_adjacency_batched(A,H,args.p,1000000)
        # A,_=add_sim(args,U,A,1000000)

        norm_A = normalize_adj_from_tensor(A, 'sym', True)

        # H=torch.linalg.svd(norm_A, full_matrices=False)
        # H=H[:,:args.dim]
        # norm_A=sparse_coo_tensor_to_coo_matrix(norm_A)
        # A=A.to_dense()
        H=top_k_eigh(norm_A,args.dim)

        # print(H.shape)
        # H=sub_randomized_svd(norm_A,n_components=n_class)
        # H=H.cpu().numpy()
        # Y=SNEM_rounding(H,t=2)
        # print(Y)
        # H,_,_=randomized_svd(norm_A,n_components=args.dim)
        # Y=torch.tensor(Y).to(device).float()
        H=torch.tensor(H).to(device).float()
        # has_inf=torch.isinf(H).any().item()
        # has_nan=torch.isnan(H).any().item()
        # print("nan:{},inf:{}".format(has_nan,has_inf))
        # exit()
        H=l_norm(H,1)
        # print(H)
        if i==args.n_time-1:
            print(omega)
        for j in range(num_view):
            omega[j]=compu_omega(H,adjs_I_minus[j],AF_norm[j],args,incis[j])
        omega=norm_w(omega)


    return H
def to_sp_csr(sparse_mat):
    import scipy.sparse as sp
    indices = sparse_mat._indices().cpu().numpy()
    values = sparse_mat._values().cpu().numpy()
    shape = sparse_mat.size()
    coo_matrix = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)

    csr_matrix = coo_matrix.tocsr()
    return csr_matrix


def to_sp_tensor(coo_dict, dtype=torch.float32, device="cpu"):


    rows = np.asarray(coo_dict['rows'])
    cols = np.asarray(coo_dict['cols'])
    values = np.asarray(coo_dict['values'])

    indices = np.stack([rows, cols], axis=0)

    indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    values_tensor = torch.tensor(values, dtype=dtype, device=device)

    sparse_tensor = torch.sparse_coo_tensor(
        indices=indices_tensor,
        values=values_tensor,
        size=coo_dict['shape'],
        device=device
    )

    return sparse_tensor
def lu_inv(adj):
    from scipy.sparse.linalg import splu
    import scipy.sparse as sp
    lu = splu(adj, permc_spec="COLAMD")  # 使用列近似最小度排序

    # 步骤 3: 生成单位矩阵的稀疏逆（按需逐列求解）
    n = adj.shape[0]
    inv_columns = []
    for i in range(n):
        e = sp.csc_matrix(([1.0], ([i], [0])), shape=(n, 1))  # 第 i 列的单位向量
        inv_col = lu.solve(e.toarray())  # 稀疏求解
        inv_columns.append(inv_col)

    inv_sparse_scipy = sp.hstack(inv_columns).tocoo()
    return inv_sparse_scipy
def filter_inv(adj,X,alpha,adj_I):
    device=adj.device
    para1=1/(alpha+1)
    para2=alpha/(alpha+1)

    inv_adj=torch.inverse((adj_I-para2*adj).to_dense())
    # aa=adj_I-para2*adj
    # aa=to_sp_csr(aa)
    # inv_adj=lu_inv(aa)
    # inv_adj=to_sp_tensor(inv_adj).to(device)
    # inv_adj=inv_adj.to_sparse()
    H=para1*safe_sparse_dot(inv_adj,X)

    return H
def compu_omega_inv(H,adj,Af,args):

    H1=safe_sparse_dot(adj,H)
    trace=0

    for i in range(H.shape[1]):
        trace+=torch.sum(H.T[i,:]*H1[:,i],dim=0)

    c = args.alpha * trace + Af * args.beta
    print(args.alpha*trace,Af * args.beta)

    omega=c**(-2)

    return omega
def Brute_Force(adjs,H,args,n_class):
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
        # H=col_norm(H,2)
        if i==0:
            H=l_norm(H,2)
        A = torch.sparse_coo_tensor(empty_indices, empty_values, shape, device=device)
        for j in range(num_view):
            weighted_adj = adjs[j] * (omega[j])
            A = A + weighted_adj
        A = A.coalesce()
        # A=0.5*adjs[0]+0.5*adjs[1]
        # U_c,U = sub_randomized_svd(H, n_components=args.hidden_dim)

        if i==args.n_time-1:
            print(omega)
        # A=reweight_adjacency_batched(A,H,args.p,1000000)
        # A,_=add_sim(args,U,A,1000000)

        norm_A = normalize_adj_from_tensor(A, 'sym', True)


        H=filter_inv(norm_A,X,args.alpha,adj_I)

        H=l_norm(H,2)
        # print(H)
        for j in range(num_view):
            omega[j] = compu_omega_inv(H,adjs_I_minus[j],AF_norm[j],args)
        omega=norm_w(omega)

    return H