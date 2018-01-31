import numpy as np
from sklearn.neighbors import NearestNeighbors as knn


class Metrics:
    
    def __init__(self, nbrs_input, Y, input_k=50, max_k=400):
        self.nbrs_input = nbrs_input
        self.n = nbrs_input.shape[0]
        self.input_k = input_k
        self.max_k = max_k
        _, self.nbrs_output = knn(n_neighbors=self.n).fit(Y).kneighbors(Y)
        
        self.rank_input = np.zeros_like(self.nbrs_input)
        self.rank_output = np.zeros_like(self.nbrs_input)
#         print(self.rank_input.shape, self.rank_output.shape, self.nbrs_input.shape, self.nbrs_output.shape)
        for i in range(self.n):
            for j in range(self.n):
                self.rank_input[i][self.nbrs_input[i, j]] = j
            for j in range(self.n):
                self.rank_output[i][self.nbrs_output[i, j]] = j
    
    def roc_metrics(self):
        n, input_k, max_k = self.n, self.input_k, self.max_k
        
        tp_rates = []
        fp_rates = []
        k_values = np.logspace(0, np.log10(500), dtype=np.int32)
        for k in k_values:
            tp_rate = 0
            fp_rate = 0
            for i in range(n):
                n_tp = np.intersect1d(self.nbrs_input[i][:input_k], self.nbrs_output[i][:k]).shape[0]
                tp_rate += n_tp / input_k
                fp_rate += (k - n_tp) / (n - input_k)
            tp_rates.append(tp_rate / n)
            fp_rates.append(fp_rate / n)
        return tp_rates, fp_rates
    
    def mean_precision_recall(self):
        n, input_k, max_k = self.n, self.input_k, self.max_k
        
        recalls = []
        precisions = []
        k_values = np.logspace(0, np.log10(500), dtype=np.int32)
        for k in k_values:
            recall = 0
            precision = 0
            for i in range(n):
                n_tp = np.intersect1d(self.nbrs_input[i][:input_k], self.nbrs_output[i][:k]).shape[0]
                recall += n_tp / input_k
                precision += n_tp / k
            recalls.append(recall / n)
            precisions.append(precision / n)
        return precisions, recalls
    
    def trustworthiness_continuity(self):
        n, input_k, max_k = self.n, self.input_k, self.max_k
        
        trusts = []
        conts = []
        k_values = np.logspace(0, np.log10(500), dtype=np.int32)
        for k in k_values:
            trust = 0
            cont = 0
            for i in range(n):
                false_pos = np.setdiff1d(self.nbrs_output[i][:k], self.nbrs_input[i][:k])
                for j in false_pos:
                    r = self.rank_input[i, j]
                    assert r >= k
                    trust += r - k
                false_neg = np.setdiff1d(self.nbrs_input[i][:k], self.nbrs_output[i][:k])
                for j in false_neg:
                    r_hat = self.rank_output[i, j]
                    assert r_hat >= k
                    cont += r_hat - k
            
            scale = 2 / (n*k * (2*n - 3*k - 1))
            trusts.append(1 - scale * trust)
            conts.append(1 - scale * cont)
                        
        return trusts, conts
