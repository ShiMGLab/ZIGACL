## run environment

python --- 3.8

pytorch --- 1.11.0

torchvision --- 1.12.0

torchaudio --- 0.11.0

scanpy --- 1.8.2

scipy --- 1.6.2

numpy --- 1.19.5

leidenalg --- 0.8.10

## file Description

["loss"]: loss fuction and refer to ["ZAGCL"]

["model"]: Containing model framwork of ZAGCL and the framwork the GAT module.

["ZAGCL.py"]: To perform model training and clustering analysis.

["train.py"]: Containing the train and clustering fuction.

["utils.py"]: Containing some data processing, cell-cell graph construction, and visualization utilities.



## Usage
For applying ZAGCL, the convenient way is  run ["ZAGCL.py"]
If you want to calculate the similarity between the predicted clustering resluts and the true cell labels (based on NMI or ARI score), please transmit your true labels into the "adata.obs['celltype']" and setting the argument "-celltype" to **True**.

argument:

    "-resolution": default: 1.0. Description: The resolution of Leiden. Advised settings in 0.1 to 1.0. 
    
    "-n_hvg": default: 2500. Description: The number of highly variable genes. In general values should be in the range 500 to 3000. 

    "-knn": default: False. Description: If True, use a hard threshold to restrict the number of neighbors to n_neighbors, that is, 
                                        consider a knn graph. Otherwise, use a Gaussian Kernel to assign low weights to neighbors more 
                                        distant than the n_neighbors nearest neighbor.
    
    "-n_neighbors": default: 15. Description: The size of local neighborhood (in terms of number of neighboring data points) used 
                                    for manifold approximation. Larger values result in more global views of the manifold, while 
                                    smaller values result in more local data being preserved. In general values should be in the 
                                    range 2 to 100. default:15
    
other arguments:

    "-celltype": default: "known". Description: The true labels of datasets are placed in adata.obs["celltype"] for model evaluation.
    
    "-save_pred_label":default: False. Description: To choose whether saves the pred_label to the dict "./pred_label"
    
    "-save_model_para":default: False. Description: To choose whether saves the model parameters to the dict "./model_save"
    
    "-save_embedding":default: True. Description: To choose whether saves the cell embedding to the dict "./embedding"
    
