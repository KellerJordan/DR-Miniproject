import numpy as np
from sklearn.neighbors import NearestNeighbors as knn


def roc_metrics(nbrs_input, nbrs_output, input_k=50):
    n = nbrs_input.shape[0]

    tp_list = []
    fp_list = []

    for k in range(1, 100):
        tp_rate = 0
        fp_rate = 0
        for i in range(n):
            n_tp = np.intersect1d(nbrs_input[i], nbrs_output[i][:k]).shape[0]
            tp_rate += n_tp / input_k
            fp_rate += (k - n_tp) / (n - input_k)
        tp_list.append(tp_rate / n)
        fp_list.append(fp_rate / n)
        
    return tp_list, fp_list

def mean_precision_recall(nbrs_input, nbrs_output, input_k=50):
    n = nbrs_input.shape[0]
    
    recall_list = []
    precision_list = []

    for k in range(1, 100):
        recall = 0
        precision = 0
        for i in range(n):
            n_tp = np.intersect1d(nbrs_input[i], nbrs_output[i][:k]).shape[0]
            recall += n_tp / input_k
            precision += n_tp / k
        recall_list.append(recall / n)
        precision_list.append(precision / n)
        
    return recall_list, precision_list
