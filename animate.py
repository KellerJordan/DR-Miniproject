import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import torch
from torchvision import datasets, transforms

import argparse
import sys
import pickle

from trimap import Wrapper as TriMap


def save_viz(dataset='mnist2500', optim='sgd', lr=None, animated=True):
    
    if dataset == 'mnist2500':
        X = np.loadtxt('data/%s_X.txt' % dataset)
        labels = np.loadtxt('data/%s_labels.txt' % dataset)
    elif dataset == 'mnist60000':
        tsfm = transforms.Compose([transforms.ToTensor()])
        mnist = datasets.MNIST('data', train=True, download=True, transform=tsfm)
        X = (mnist.train_data.type(torch.FloatTensor) / 255.0).view(-1, 784).numpy()
        labels = mnist.train_labels.numpy()
    
    print('Computing TriMap embedding using %s optimizer...' % optim)

    trimap = TriMap(X)
    
    triplets_path = 'models/%s_triplets.pkl' % dataset
    
    try:
        with open(triplets_path, 'rb') as f:
            triplets_init = pickle.load(f)
            
        trimap.triplets, trimap.weights = triplets_init
    except:
        trimap.generate_triplets(verbose=True)
        triplets_state = (trimap.triplets, trimap.weights)
        
        with open(triplets_path, 'wb') as f:
            pickle.dump(triplets_state, f)
        
    Y_seq = trimap.embed(num_iters=1000, optimizer=optim, return_seq=True, verbose=True)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [], 20, [])
    ax.set_title('%s (epoch 0)' % optim)

    def init():
        return scatter,

    def update(i):
        if i % 50 == 0:
            print('Animating, iteration %d / %d' % (i, len(Y_seq)))
        ax.clear()
        ax.scatter(Y_seq[i][:, 0], Y_seq[i][:, 1], 20, labels)
        ax.set_title('%s (epoch %d)' % (optim, i))
        return ax, scatter
    
    if animated:
        anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(Y_seq), interval=50)
        path = 'animations/%s.gif' % optim
        print('Saving animation as %s' % path)
        anim.save(path, writer='imagemagick', fps=30)
    
    else:
        Y = Y_seq[-1]
        ax.clear()
        ax.scatter(Y[:, 0], Y[:, 1], 20, labels)
        ax.set_title('%s' % optim)
        path = 'animations/%s.png' % optim
        print('Saving figure as %s' % path)
        plt.savefig(path)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist2500')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=1.0)
    args = parser.parse_args()
    save_viz(args.dataset, args.optimizer, args.lr)
