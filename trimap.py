"""
pytorch trimap implementation
@author Keller Jordan, kjjordan@ucsc.edu
"""

from sklearn.neighbors import NearestNeighbors as knn
import numpy as np
import pickle


import torch
from torch import nn, optim
from torch.autograd import Variable


class TriMapper(nn.Module):
    
    def __init__(self, triplets, weights, out_shape):
        super(TriMapper, self).__init__()
        n, num_dims = out_shape
        self.Y = nn.Embedding(n, num_dims, sparse=False)
        for W in self.Y.parameters():
            W.data = torch.Tensor(np.random.normal(size=[n, num_dims]) * 0.0001)
        self.triplets = Variable(torch.cuda.LongTensor(triplets))
        self.weights = Variable(torch.cuda.FloatTensor(weights))
    
    def forward(self):
        y_ij = self.Y(self.triplets[:, 0]) - self.Y(self.triplets[:, 1])
        y_ik = self.Y(self.triplets[:, 0]) - self.Y(self.triplets[:, 2])
        d_ij = 1 + torch.sum(y_ij**2, -1)
        d_ik = 1 + torch.sum(y_ik**2, -1)
        num_viol = torch.sum((d_ij > d_ik).type(torch.FloatTensor))
        loss = self.weights.dot(d_ij / (d_ij + d_ik))
        return loss, num_viol
    
    def get_embeddings(self):
        return self.Y._parameters['weight'].cpu().data.numpy()


class Wrapper:
    
    def __init__(self, X):
        self.X = X
        
    def embed(self, num_iters=2000, optimizer='sgd', lr=None,
              return_seq=False, verbose=False):

        num_examples = self.X.shape[0]
        num_triplets = self.triplets.shape[0]
        model = TriMapper(self.triplets, self.weights, out_shape=[num_examples, 2])
        model.cuda()

        tol = 1e-7
        C = np.inf

        if lr == None:
            eta = 1000.0 / num_triplets * num_examples

        if optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=eta)
        elif optimizer == 'sgd-momentum':
            optimizer = optim.SGD(model.parameters(), lr=eta, momentum=.9)
        elif optimizer == 'adam':
            optimizer = optim.Adam(model.parameters())
        elif optimizer == 'adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=eta)
        elif optimizer == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters())

        if return_seq:
            Y_seq = []

        for i in range(num_iters):
            old_C = C

            loss, num_viol = model()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            C = loss.data.cpu().numpy()
            viol = float(num_viol) / num_triplets

            if optimizer in ['sgd', 'sgd-momentum']:
                if old_C > C + tol:
                    eta = eta * 1.01
                else:
                    eta = eta * 0.5
                optimizer.param_groups[0]['lr'] = eta

            if return_seq:
                Y = model.get_embeddings()
                Y_seq.append(Y)

            if verbose and (i+1) % 100 == 0:
                print('Iteration: %4d, Loss: %3.3f, Violated triplets: %0.4f' % (i+1, loss, viol))

        Y = model.get_embeddings()

        return Y_seq if return_seq else Y
    
    def generate_triplets(self, kin=50, kout=10, kr=5,
                          weight_adj=False, random_triplets=True, verbose=False):
        """
        generate_triplets()
        Created on Sat May 27 12:46:25 2017

        @author: ehsanamid
        """
        
        X = self.X
        
        num_extra = np.maximum(kin+50, 60) # look up more neighbors
        # ^ ???
        n = X.shape[0]
        nbrs = knn(n_neighbors= num_extra + 1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)
        sig = np.maximum(np.mean(distances[:, 10:20], axis=1), 1e-20) # scale parameter
        P = np.exp(-distances**2/np.reshape(sig[indices.flatten()],[n, num_extra + 1])/sig[:, np.newaxis])
        sort_indices = np.argsort(-P, axis = 1) # actual neighbors

        triplets = np.zeros([n * kin * kout, 3], dtype=np.int32)
        weights = np.zeros(n * kin * kout)

        cnt = 0
        for i in range(n):
            for j in range(kin):
                sim = indices[i,sort_indices[i, j+1]]
                p_sim = P[i,sort_indices[i, j+1]]
                rem = indices[i,sort_indices[i, :j+2]].tolist()
                l = 0
                while (l < kout):
                    out = np.random.choice(n)
                    if out not in rem:
                        triplets[cnt] = [i, sim, out]
                        p_out = max(np.exp(-np.sum((X[i] - X[out])**2) / (sig[i] * sig[out])), 1e-20)
                        weights[cnt] = p_sim / p_out
                        rem.append(out)
                        l += 1
                        cnt += 1
            if verbose and (i+1) % 500 == 0:
                print('Generated triplets %d / %d' % (i+1, n))
        if random_triplets:
    #         kr = 5
            triplets_rand = np.zeros([n * kr, 3])
            weights_rand = np.zeros(n * kr)
            for i in range(n):
                cnt = 0
                while cnt < kr:
                    sim = np.random.choice(n)
                    out = np.random.choice(n)
                    if sim == i or out == i or out == sim:
                        continue
                    p_sim = max(np.exp(-np.sum((X[i]-X[sim])**2)/(sig[i] * sig[sim])), 1e-20)
                    p_out = max(np.exp(-np.sum((X[i]-X[out])**2)/(sig[i] * sig[out])), 1e-20)
                    if p_sim < p_out:
                        sim, out = out, sim
                        p_sim, p_out = p_out, p_sim
                    triplets_rand[i * kr + cnt] = [i, sim, out]
                    weights_rand[i * kr + cnt] = p_sim / p_out
                    cnt += 1
                if verbose and (i+1) % 500 == 0:
                    print('Generated random triplets %d / %d' % (i+1, n))
            triplets = np.vstack((triplets, triplets_rand))
            weights = np.hstack((weights, weights_rand))
        triplets = triplets[~np.isnan(weights)]
        weights = weights[~np.isnan(weights)]
        weights /= np.max(weights)
        weights += 0.0001
        if weight_adj:
            weights = np.log(1 + 50 * weights)
            weights /= np.max(weights)
        
        self.triplets = triplets
        self.weights = weights
        # do some pickling
