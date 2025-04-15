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


def clustering_f1(y_true, y_pred, average='weighted'):
    """
    使用多数投票法计算聚类 F1 值.

    Args:
        y_true: 真实标签数组.
        y_pred: 预测的聚类标签数组.
        average: F1 值的平均方法 ('micro', 'macro', 'weighted', 'None').

    Returns:
        F1 值 (浮点数).
    """
    from sklearn.metrics import f1_score, confusion_matrix
    from collections import Counter,defaultdict
    # 构建簇到真实标签的映射
    cluster_mapping = defaultdict(list)
    for i, cluster_id in enumerate(y_pred):
        cluster_mapping[cluster_id].append(y_true[i])

    # 使用多数投票为每个簇分配一个真实标签
    predicted_labels = np.zeros_like(y_pred)
    for cluster_id, true_labels in cluster_mapping.items():
        most_common_label = Counter(true_labels).most_common(1)[0][0]
        predicted_labels[y_pred == cluster_id] = most_common_label

    # 计算 F1 值
    f1 = f1_score(y_true, predicted_labels, average=average, labels=np.unique(y_true))
    return f1
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
# def threshold_edges(edge_index, threshold):
#
#     node_counts = torch.zeros(edge_index.max() + 1, dtype=torch.int)
#     node_counts.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0]))
#
#     nodes_to_threshold = torch.nonzero(node_counts > threshold).squeeze()
#
#     mask = torch.ones_like(edge_index[0], dtype=torch.bool)
#     for node in nodes_to_threshold:
#         node_edges = torch.nonzero(edge_index[0] == node).squeeze()
#         mask[node_edges[threshold:]] = False
#
#     edge_index = edge_index[:, mask]
#
#     return edge_index
# def generate_random_bool_vector(length, num_true):
#     """Generates a boolean vector with a specified length and number of True values.
#
#     Args:
#         length: The total length of the vector.
#         num_true: The number of True values in the vector.
#
#     Returns:
#         A torch.BoolTensor with the specified properties, or None if the input is invalid.
#     """
#
#     if num_true < 0 or num_true > length:
#         return None  # Handle invalid input
#
#     # Create a vector of indices and shuffle them
#     # indices = torch.arange(length)
#     shuffled_indices = torch.randperm(length)
#
#     # Create the boolean vector
#     bool_vector = torch.zeros(length, dtype=torch.bool)
#     bool_vector[shuffled_indices[:num_true]] = True
#
#     return bool_vector
def get_high_degree_nodes(degree, threshold):

    high_d_nodes = torch.where(degree > threshold)[0]
    return high_d_nodes
# def no_self_loop(tensor):
#
#     return not torch.all(tensor[:, 0] == tensor[:, -1])
# def mask_subgraph_edges(high_d_nodes, subgraph_edge,subgraph_values):
#
#     src_in_high_d_nodes = torch.isin(subgraph_edge[0], high_d_nodes)
#     dst_in_high_d_nodes=torch.isin(subgraph_edge[1], high_d_nodes)
#     mask_index = torch.logical_and(src_in_high_d_nodes, dst_in_high_d_nodes)
#     mask_self_loop=no_self_loop(subgraph_edge)
#     mask_index=mask_index&mask_self_loop
#     masked_subgraph_edge = subgraph_edge[:,~mask_index]
#     masked_subgraph_values = subgraph_values[~mask_index]
#     return masked_subgraph_edge,masked_subgraph_values
def sample_subgraph(adj_all, args, center_nodes, num_hops,high_d_nodes):
    device=adj_all.device
    current_nodes = torch.unique(center_nodes)
    all_nodes = current_nodes.clone()

    edges_at_k_minus_1 = None
    # current_graph=None

    for i in range(num_hops):
        mask = torch.isin(adj_all._indices()[0], current_nodes)
        # if i>=(num_hops // 2):
        sampled_edges = adj_all._indices()[:, mask]

        tar_nodes = torch.unique(sampled_edges[1])
        mask_same_node=torch.isin(tar_nodes, current_nodes)
        current_nodes=tar_nodes[~mask_same_node]

        mask=torch.isin(current_nodes,high_d_nodes)
        current_nodes = current_nodes[~mask]
        all_nodes = torch.unique(torch.cat([all_nodes, current_nodes]))

        if i == (num_hops // 2 - 1):
            # edges_at_k_minus_1 = current_graph.clone()
            nodes_at_k_minus_1=all_nodes.clone()

    mask = torch.isin(adj_all._indices()[0], all_nodes) & torch.isin(adj_all._indices()[1], all_nodes)
    subgraph_edges = adj_all._indices()[:, mask]

    subgraph_values = adj_all._values()[mask]

    # subgraph_edges,subgraph_values=mask_subgraph_edges(high_d_nodes,subgraph_edges,subgraph_values)

    # map_start=time.time()
    # node_mapping_array = np.zeros(max(all_nodes) + 1, dtype=int)
    # for idx, node in enumerate(all_nodes):
    #     node_mapping_array[node.item()] = idx
    # node_mapping = torch.tensor(node_mapping_array, device=device)
    # node_mapping = {node.item(): idx for idx, node in enumerate(all_nodes)}
    # src = subgraph_edges[0].to(device)
    # src = node_mapping[src]
    #
    # dst = subgraph_edges[1].to(device)
    # dst = node_mapping[dst]
    # sizes=len(all_nodes)
    # indices = torch.stack((src, dst), dim=0).to(device)
    # end_map=time.time()-map_start
    # print("Elapsed time:{}".format(end_map))
    subgraph = torch.sparse_coo_tensor(subgraph_edges, subgraph_values,adj_all.size()).to(device)
    # subgraph=subgraph+subgraph.transpose(0,1)
    # subgraph=remove_self_loop(subgraph)
    # subgraph=sparse_tensor_add_self_loop(subgraph)
    # inverse_node_mapping = torch.tensor(all_nodes, device=device)

    # nodes_at_k_minus_1_mapped = node_mapping[nodes_at_k_minus_1]
    # subgraph_features = features[all_nodes]
    print(subgraph._nnz())
    return subgraph, nodes_at_k_minus_1
def sparse_tensor_add_minus_self_loop(adj):

    node_num = adj.shape[0]
    index = torch.stack((torch.tensor(range(node_num)), torch.tensor(range(node_num))), dim=0)
    values = torch.ones(node_num)*(1e-15)

    adj_new = torch.sparse.FloatTensor(torch.cat((index, adj.indices()), dim=1), torch.cat((values, adj.values()),dim=0), adj.shape)
    return adj_new.coalesce()
def normalize_subadj(adj):
    adj = adj.coalesce()
    new_adj=sparse_tensor_add_minus_self_loop(adj)
    inv_sqrt_degree=1. / (torch.sqrt(torch.sparse.sum(new_adj, dim=1).values()))
    D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
    new_values = adj.values() * D_value

    return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())
def compute_degree(adj):
    new_indices=adj._indices()

    new_values=torch.ones(new_indices.shape[1])
    new_adj=torch.sparse_coo_tensor(new_indices, new_values, adj.size())
    degree=torch.sparse.sum(new_adj,dim=1).values()

    return degree





# def sub_randomized_svd(
#     Z,
#     n_components,
#     *,
#     n_iter="auto",
#     flip_sign=True,
#     random_state="warn",
#     n_oversamples=10,
# ):
#     # if isinstance(Z, torch.Tensor) and (Z.is_sparse_coo() or Z.is_sparse_csr()):
#     #     warnings.warn(
#     #         "SVD with sparse PyTorch tensors may not perform efficiently. "
#     #         "Consider converting to a dense tensor for better performance.",
#     #         UserWarning,
#     #     )
#
#     if random_state == "warn":
#         warnings.warn(
#             "The default random_state will change in the future. "
#             "Set it to None or a fixed integer to avoid future warnings.",
#             FutureWarning,
#         )
#         random_state = 0
#
#     random_state = check_random_state(random_state)
#     n_random = n_components + n_oversamples
#
#     if n_iter == "auto":
#         n_iter = 7 if n_components < 0.1 * min(Z.shape) else 4
#     seed_value = 42
#     generator = torch.Generator(device=Z.device)
#     generator.manual_seed(seed_value)
#     G = torch.randn(Z.size(1), n_random, device=Z.device, dtype=Z.dtype,generator=generator)
#
#     Q = G
#     # print(Z,Q)
#     for _ in range(n_iter):
#         Q = safe_sparse_dot(Z, Q)
#         Q = safe_sparse_dot(Z.t(), Q)
#         Q=l_norm(Q,1)
#     Q = safe_sparse_dot(Z, Q)
#     Q = l_norm(Q, 1)
#     Q, _ = torch.linalg.qr(Q, mode='reduced')
#
#     B = safe_sparse_dot(Z.t(), Q).t()
#     try:
#         Uhat, s, Vt = torch.linalg.svd(B, full_matrices=False)
#     except RuntimeError:
#         Vt, s, Uhat = torch.linalg.svd(B.t(), full_matrices=False)
#         Uhat = Vt  # Adjust based on matrix orientation
#
#     U = torch.matmul(Q, Uhat)
#
#     if flip_sign:
#         U, Vt = svd_flip(U, Vt)
#     U_c=U@torch.diag(s)
#     return U_c[:, 1:n_components+1],U[:,1:n_components]
def get_I(n,device):

    src=torch.arange(n)
    dst=torch.arange(n)
    indices=torch.stack((src,dst))
    values=torch.ones(n)
    adj_I=torch.sparse_coo_tensor(indices,values,(n,n)).to(device)
    return adj_I
# def compu_omega(H,adj,Af,args,incis):
#
#     # H1=safe_sparse_dot(adj,H)
#     # trace=0
#     #
#     # for i in range(H.shape[1]):
#     #     trace+=torch.sum(H.T[i,:]*H1[:,i],dim=0)
#     device=H.device
#     incis=incis.to(device)
#     H_m = torch.matmul(H.T, incis)
#     # print(Af,F_norm(H_m)**2)
#     trace = F_norm(H_m) ** 2
#     # print(Af,trace)
#     c=torch.sqrt(Af)*args.beta+args.sigma*trace
#     omega=c**(-2)
#     # print(trace)
#     # omega=trace**(-1/2)
#     # print(H,trace)
#     return omega

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
        inci=cwt(inc.T,args.k[i],seed=42)# get E'
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
        # H=col_norm(H,2)
        # if i==0:
        #     H=l_norm(H,2)
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
        # H=filter_double_para(norm_A,X,args.n_T,args.alpha,args.sigma)
        # H=l_norm(H,2)
        # H=torch.linalg.svd(norm_A, full_matrices=False)
        # H=H[:,:args.dim]
        # norm_A=sparse_coo_tensor_to_coo_matrix(norm_A)
        # A=A.to_dense()
        H=top_k_eigh(norm_A,args.dim,device)

        # print(H.shape)
        # H=sub_randomized_svd(norm_A,n_components=n_class)
        # H=H.cpu().numpy()
        # Y=SNEM_rounding(H,t=2)
        # print(Y)
        # H,_,_=randomized_svd(norm_A,n_components=args.dim)
        # Y=torch.tensor(Y).to(device).float()
        H=torch.tensor(H).to(device).float()

        # H=l_norm(H,2)
        # print(H)
        if i==args.n_time-1:
            print(omega)
        for j in range(num_view):
            omega[j]=compu_omega(H,adjs_I_minus[j],AF_norm[j],args,incis[j])
        omega=norm_w(omega)


    return H