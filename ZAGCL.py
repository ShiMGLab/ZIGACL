import torch
from warnings import simplefilter 
import argparse
from sklearn import preprocessing
import random
import numpy as np
import utils
import scanpy as sc

from model import GraphEncoder


# from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from train import train, clustering, loss_func
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, davies_bouldin_score
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_z', type=int, default=16,
                        help='the number of dimension of latent vectors for each cell, default:16')
    parser.add_argument('--n_init', type=int, default=20, help="input")
    parser.add_argument('--n_hvg', type=int, default=2500,
                        help='the number of the highly variable genes, default:2500')
    parser.add_argument('--training_epoch', type=int, default=200,
                        help='epoch of train stage, default:200')
    parser.add_argument('--clustering_epoch', type=int, default=100,
                        help='epoch of clustering stage, default:100')
    parser.add_argument('--resolution', type=float, default=0.2,
                        help='''the resolution of Leiden. The smaller the settings to get the more clusters
                        , advised to 0.1-1.0, default:1.0 scDGAT=0.2''')
    parser.add_argument('--connectivity_methods', type=str, default='gauss')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='''The size of local neighborhood (in terms of number of neighboring data points) used 
                        for manifold approximation. Larger values result in more global views of the manifold, while 
                        smaller values result in more local data being preserved. In general values should be in the 
                        range 2 to 100. default:15''')
    parser.add_argument('--knn', type=int, default=False,
                        help='''If True, use a hard threshold to restrict the number of neighbors to n_neighbors, 
                        that is, consider a knn graph. Otherwise, use a Gaussian Kernel to assign low weights to
                        neighbors more distant than the n_neighbors nearest neighbor. default:False''')
    parser.add_argument('--name', type=str, default='Romanov',
                        help='name of input fijle(a h5ad file: Contains the raw count matrix "X")')

    # Muraro	Romanov	 Klein     Quake_10x_Bladder  Quake_10x_Limb_Muscle	    Quake_10x_Spleen 	Quake_Smart-seq2_Diaphragm    Quake_Smart-seq2_Limb_Muscle
    # Quake_Smart-seq2_Lung     Quake_Smart-seq2_Trachea        Pancreas_human1     Pancreas_human2     Pancreas_human3
    #Pancreas_human4     Pancreas_mouse

    parser.add_argument('--n_clusters', type=int, default=9,help='真實標簽聚類數，len(np.unique(dataset.y))')

    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

    parser.add_argument('--celltype', type=str, default='known',
                        help='the true labels of datasets are placed in adata.obs["celltype"]')
    parser.add_argument('--save_pred_label', type=str, default=False,
                        help='To choose whether saves the pred_label to the dict "./pred_label"')
    parser.add_argument('--save_model_para', type=str, default=False,
                        help='To choose whether saves the model parameters to the dict "./model_save"')
    parser.add_argument('--save_embedding', type=str, default=False,
                        help='To choose whether saves the cell embedding to the dict "./embedding"')
    parser.add_argument('--save_umap', type=str, default=False,
                        help=' True To choose whether saves the visualization to the dict "./umap_figure"')
    parser.add_argument('--max_num_cell', type=int, default=10000)

    parser.add_argument('--cuda', type=bool, default=False,
                        help='use GPU, or else use cpu (setting as "False")')
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    simplefilter(action='ignore', category=FutureWarning)
    
    random.seed(1)
    if args.save_umap is True:
        umap_save_path = ['./umap_figure/%s_pred_label.png'%(args.name),'./umap_figure/%s_true_label.png'%(args.name)]
    else:
        umap_save_path = [None, None]

    adata, rawData, dataset, adj, r_adj = utils.load_data('./Data/AnnData/{}'.format(args.name),args=args)

    unique_celltypes = adata.obs['celltype'].unique()

    num_celltypes = len(unique_celltypes)

    args.n_clusters = num_celltypes

    print(f"Unique cell types: {unique_celltypes}")
    print(f"Number of cell types: {num_celltypes}")
    
    if args.celltype == "known":  
        celltype = adata.obs['celltype'].tolist()
    else:
        celltype = None

    if adata.shape[0] < args.max_num_cell:

        size_factor = adata.obs['size_factors'].values
        Zscore_data = preprocessing.scale(dataset)    

        args.n_input = dataset.shape[1]
        print(args)

        t = 10
        tran_prob = normalize(adj, norm="l1", axis=0)
        M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
        M = torch.Tensor(M_numpy).to(device)

        init_model = GraphEncoder(256, 64, 64, 256, n_input=args.n_input, n_z=args.n_z, alpha=args.alpha, n_clusters=args.n_clusters, device=device)
        pretrain_model, _ = train(M, init_model, Zscore_data, rawData, adj, size_factor, device, args)
        metric, pred_label, model, _ = clustering(M, pretrain_model, Zscore_data, rawData, celltype,
                                                  adj, size_factor, device, args)

        asw = metric[0]
        db  = metric[1]
        if celltype is not None:
            nmi = np.round(normalized_mutual_info_score(celltype, pred_label), 3)
            ari = np.round(adjusted_rand_score(celltype, pred_label), 3)
            print("Final ASW %.3f, DB %.3f, ARI %.3f, NMI %.3f"% (asw, db, ari, nmi))
            
        else:
            print("Final ASW %.3f, DB %.3f"% (asw, db))

        print(args.name)

        data = torch.Tensor(Zscore_data).to(device)

        with torch.no_grad():
            z, pi, mean ,disp, pred, _, _ = model(data, adj, M)
            if args.save_umap is True:
                # utils.umap_visual(z.detach().cpu().numpy(),
                #                   label = pred_label,
                #                   title='predicted label',
                #                   save_path = umap_save_path[0],
                #                   asw_used=True)
                utils.umap_visual(z.detach().cpu().numpy(),
                                  label = pred_label,
                                  title='predicted label',
                                  save_path = umap_save_path[0],
                                  asw_used=None)
                if args.celltype == "known":
                    utils.umap_visual(z.detach().cpu().numpy(),
                                      label = celltype,
                                      title='true label',
                                      save_path = umap_save_path[1])
        if args.save_embedding is True:
            np.savetxt('./embedding/%s.csv'%(args.name), z.detach().cpu().numpy())
        if args.save_pred_label is True:
            np.savetxt('./pred_label/%s.csv'%(args.name),pred_label)
        if args.save_model_para is True:
            torch.save(model.state_dict(), './model_save/%s.pkl'%(args.name))
