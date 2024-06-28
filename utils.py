from sklearn.preprocessing import LabelEncoder
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
from collections import Counter
import torch
import random
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import gudhi

from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.utils import  from_scipy_sparse_matrix
import scipy.sparse as sp

import seaborn as sns
import matplotlib.pyplot as plt

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

def graph_alpha(spatial_locs, n_neighbors=10):
    """
    Construct a geometry-aware spatial proximity graph of the spatial spots of cells by using alpha complex.
    :param adata: the annData object for spatial transcriptomics data with adata.obsm['spatial'] set to be the spatial locations.
    :type adata: class:`anndata.annData`
    :param n_neighbors: the number of nearest neighbors for building spatial neighbor graph based on Alpha Complex
    :type n_neighbors: int, optional, default: 10
    :return: a spatial neighbor graph
    :rtype: class:`scipy.sparse.csr_matrix`
    """
    A_knn = kneighbors_graph(spatial_locs, n_neighbors=n_neighbors, mode='distance')
    estimated_graph_cut = A_knn.sum() / float(A_knn.count_nonzero())
    spatial_locs_list = spatial_locs.tolist()
    n_node = len(spatial_locs_list)
    alpha_complex = gudhi.AlphaComplex(points=spatial_locs_list)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=estimated_graph_cut ** 2)
    skeleton = simplex_tree.get_skeleton(1)
    initial_graph = nx.Graph()
    initial_graph.add_nodes_from([i for i in range(n_node)])
    for s in skeleton:
        if len(s[0]) == 2:
            initial_graph.add_edge(s[0][0], s[0][1])

    extended_graph = nx.Graph()
    extended_graph.add_nodes_from(initial_graph)
    extended_graph.add_edges_from(initial_graph.edges)

    # Remove self edges
    for i in range(n_node):
        try:
            extended_graph.remove_edge(i, i)
        except:
            pass

    return nx.to_scipy_sparse_matrix(extended_graph, format='csr')


def adata_knn1(adata, method, knn, n_neighbors, metric='cosine'):
    if adata.shape[0] >=10000:
        sc.pp.pca(adata, n_comps=50)
        n_pcs = 50
    else:
        n_pcs=0
    if method == 'umap':
        sc.pp.neighbors(adata, method = method, metric=metric,
                            knn=knn, n_pcs=n_pcs, n_neighbors=n_neighbors)
        r_adj = adata.obsp['distances']
        adj = adata.obsp['connectivities']
    elif method == 'gauss':
        sc.pp.neighbors(adata, method = 'gauss', metric=metric,
                            knn=knn, n_pcs=n_pcs, n_neighbors=n_neighbors)
        r_adj = adata.obsp['distances']
        adj = adata.obsp['connectivities']

    return adj, r_adj

#constructing the cell-cell graph
def adata_knn(adata, method, knn, n_neighbors, metric='cosine'):
    if adata.shape[0] >=10000:
        sc.pp.pca(adata, n_comps=50)
        n_pcs = 50
    else:
        n_pcs=0

    distances_csr_matrix = \
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, knn=True, copy=True).obsp[
            'distances']
    # ndarray
    distances = distances_csr_matrix.A

    sigma = 1.0  # Adjust as needed

    # Compute similarity using the Gaussian kernel
    similarities = np.exp(-distances ** 2 / (2 * sigma ** 2))

    # Convert similarities to an adjacency matrix
    adjacency_matrix = sp.csr_matrix(similarities)

    # Optional: Normalize the adjacency matrix (if required by GCNII)
    normalized_adjacency = normalize(adjacency_matrix)

    # Optional: Convert to PyTorch sparse tensor (if required by GCNII)
    adj_tensor = sparse_mx_to_torch_sparse_tensor(normalized_adjacency)

    # resize
    neighbors = np.resize(distances_csr_matrix.indices, new_shape=(distances.shape[0], n_neighbors))

    cutoff = np.mean(np.nonzero(distances), axis=None) + float(0.0) * np.std(
        np.nonzero(distances), axis=None)

    # shape: 2 * (the number of edge)
    edgelist = []
    for i in range(neighbors.shape[0]):
        for j in range(neighbors.shape[1]):
            if neighbors[i][j] != -1:
                pair = (str(i), str(neighbors[i][j]))
                if 0.0:
                    distance = distances[i][j]
                    if distance < cutoff:
                        if i != neighbors[i][j]:
                            edgelist.append(pair)
                else:
                    if i != neighbors[i][j]:
                        edgelist.append(pair)



    num_nodes = adata.shape[0]
    # print(f'Number of nodes in graph: {num_nodes}.')
    # print(f'The graph has {len(edgelist)} edges.')
    edge_index = np.array(edgelist).astype(int).T
    edge_index = to_undirected(torch.from_numpy(edge_index).to(torch.long), num_nodes)
    # print(f'The undirected graph has {edge_index.shape[1]}.')
    global_graph = Data(x=adata.X, edge_index=edge_index)
    # numbering the node
    global_graph.n_id = torch.arange(global_graph.num_nodes)



    return adj_tensor, distances_csr_matrix

# To load gene expression data file into the (pre-)train function.
def load_data(dataPath, args, metric='cosine', 
              dropout=0.4, preprocessing_sc=True):
    adata = ad.read(dataPath + '.h5ad')
    print(adata.shape)

    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.filter_genes(adata, min_cells=1)
    adata.raw = adata
    # print(adata)
    adata.X = adata.X.astype(np.float32)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)   #ZINB相關
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2500)
    adata.raw.var['highly_variable'] = adata.var['highly_variable']
    adata = adata[:, adata.var['highly_variable']]
    dataMat = adata.X
    rawData = adata.raw[:, adata.raw.var['highly_variable']].X     #Zinb
    
    if dropout !=0:
        dataMat, rawData = random_mask(dataMat, rawData, dropout)    
        adata.X = dataMat
    # Construct graph

    adj, r_adj = adata_knn(adata, method = args.connectivity_methods, knn=args.knn,
                           n_neighbors = args.n_neighbors, metric=metric)
    return adata, rawData, dataMat, adj, r_adj
   


# using Leiden algorithm to initialize the clustering centers.
def use_Leiden(features, resolution=1):
    #from https://github.com/eleozzr/desc/blob/master/desc/models/network.py line 241
    adata0=sc.AnnData(features)
    sc.pp.neighbors(adata0, knn=False, method = 'gauss', metric='cosine', n_pcs=0)
    sc.tl.leiden(adata0, resolution=0.1)
    Y_pred_init=adata0.obs['leiden']
    init_pred=np.asarray(Y_pred_init,dtype=int)
    features=pd.DataFrame(adata0.X,index=np.arange(0,adata0.shape[0]))
    Group=pd.Series(init_pred,index=np.arange(0,adata0.shape[0]),name="Group")
    Mergefeature=pd.concat([features,Group],axis=1)
    cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
    return cluster_centers, init_pred

# using spectral clustering to initialize the clustering centers.
def use_SpectralClustering(data, adj, args):
    #from https://github.com/Philyzh8/scTAG/blob/38ca65d781a20c3c058ac1d4e58f6d17aaf89908/train.py#L30 line 87
    from sklearn.cluster import SpectralClustering
    Y_pred_init = SpectralClustering(n_clusters=args.n_clusters,affinity="precomputed", 
                                     assign_labels="discretize",random_state=0).fit_predict(adj)
    init_pred=np.asarray(Y_pred_init,dtype=int)
    features=pd.DataFrame(data,index=np.arange(0,data.shape[0]))
    Group=pd.Series(init_pred,index=np.arange(0,data.shape[0]),name="Group")
    Mergefeature=pd.concat([features,Group],axis=1)
    cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
    return cluster_centers,init_pred

def random_downsimpling(data, num_cell):
    '''
    data: AnnData type data
    num_cell: The number of sampled cells.
    '''
    p = num_cell/data.shape[0]
    matrix = pd.DataFrame([np.array(range(data.shape[0])),data.obs['celltype']]).transpose()
    matrix.columns = ['index','celltype']
    sort_matrix = matrix.sort_values(['celltype','index'])
    t_sample = []
    groups = Counter(sort_matrix['celltype'])
    # i = 0
    for j in groups.values():
        sample =[]
        sub_sample = random.sample(range(j), int(j*p))
        # i += j
        for k in range(j):
            if k in sub_sample:
                sample.append(1)
            else:
                sample.append(0)
        t_sample += sample
    sort_matrix['sampling'] = np.array(t_sample,dtype=np.bool8)
    final_sort_matrix = sort_matrix.sort_values(['index'])
    sample = []
    for i in range(data.shape[0]):
        if final_sort_matrix['sampling'][i]:
            sample.append(i)
    new_X = data.X[sample,:]
    new_obs = data.obs.iloc[sample,:]
    new_data_raw_X = data.raw.X[sample,:]
    new_data = ad.AnnData(X = new_X, obs = new_obs, var = data.var)  
    new_data.raw = ad.AnnData(X = new_data_raw_X, obs = new_data.obs, var = data.raw.var)  
    return new_data
    

def random_mask(data, raw_data, p):
    '''
    Before training, the gene expression matrix and the corresponding count matrix 
    are performed same masking to get the results of figure5b
    data: gene expression matrix
    raw_data: the corresponding count matrix of data
    p: the dropout rate
    '''
    new_l =[]
    new_l2 =[]
    for i in range(data.shape[0]):
        l =[]
        l2 =[]
        rowdata = data[i,:]
        rowdata2 = raw_data[i,:]
        range_row = range(data.shape[1])
        sample = random.sample(range_row, int(data.shape[1] * (1-p)))
        #sample = random.sample(range_row, int(data.shape[1] * (1-p))) p代表的含义,它是0.2,代表dropout rate 是20%吗还是80%?
        for j in range(data.shape[1]):
            if j in sample:
                l.append(rowdata[j])
                l2.append(rowdata2[j])
            else:
                l.append(0)
                l2.append(0)
        new_l.append(np.array(l))
        new_l2.append(np.array(l2))
    new_data = np.array(new_l)
    new_rawdata = np.array(new_l2)    
    return new_data, new_rawdata


#getting predicted cell label from allocation matrix P or Q.
def dist_2_label(p):
    _, label = torch.max(p, dim=1)
    return label.data.cpu().numpy()

def umap_visual(data, title=None, save_path=None, label=None, asw_used=None):
    reducer = umap.UMAP(random_state=4132231)
    embedding = reducer.fit_transform(data)
    n_lables = len(set(label)) + 1
    mean_silhouette_score = silhouette_score(data, label)
    # ARI = calcu_ARI(label, true_label)
    # NMI = normalized_mutual_info_score(true_label, label)
    xlim_l = int(embedding[:, 0].min()) - 2
    xlim_r = int(embedding[:, 0].max()) + 2
    ylim_d = int(embedding[:, 1].min()) - 2
    ylim_u = int(embedding[:, 1].max()) + 2
    plt.figure(figsize = (6,4), dpi=200)
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap='Spectral', s=5)

    b = label
    print(label)
    label_encoder = LabelEncoder()
    num_labels = label_encoder.fit_transform(label)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=num_labels, cmap='Spectral', s=5)


    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(n_lables)).set_ticks(np.arange(n_lables))
    plt.xlim((xlim_l, xlim_r))
    plt.ylim((ylim_d, ylim_u))
    plt.title('UMAP projection of the {0}'.format(title))
    if asw_used is not None:
        plt.text(xlim_r-2, ylim_d+1.5, "ASW=%.3f"%(mean_silhouette_score),
                  ha="right",)
    plt.grid(False)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    
# def cluster_acc(y_true, y_pred):
#     """
#     Calculate clustering accuracy. Require scikit-learn installed
#     # Arguments
#         y: true labels, numpy.array with shape `(n_samples,)`
#         y_pred: predicted labels, numpy.array with shape `(n_samples,)`
#     # Return
#         accuracy, in [0,1]
#     """
#     y_true = y_true.astype(np.int64)
#     assert y_pred.size == y_true.size
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#     from scipy.optimize import linear_sum_assignment as linear_assignment
#     ind = linear_assignment(w.max() - w)
#     return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size 