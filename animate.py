import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import sys

import trimap

def record_gif(optimizer):
    
    X = np.loadtxt('data/mnist2500_X.txt')
    labels = np.loadtxt('data/mnist2500_labels.txt')
    
    print('Computing TriMap embedding using %s optimizer' % optimizer)
    Y, Y_seq = trimap.embed(X, optimizer=optimizer, return_seq=True, verbose=True)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [], 20, [])
    ax.set_title('%s (epoch 0)' % optimizer)

    def init():
        return scatter,

    def update(i):
        if i % 50 == 0:
            print('Animating, iteration %d / %d' % (i, len(Y_seq)))
        ax.clear()
        ax.scatter(Y_seq[i][:, 0], Y_seq[i][:, 1], 20, labels)
        ax.set_title('%s (epoch %d)' % (optimizer, i))
        return ax, scatter
    
    anim = FuncAnimation(fig, update, init_func=init,
                     frames=len(Y_seq), interval=50)
    path = 'animations/%s.gif' % optimizer
    print('Saving animation as %s' % path)
    anim.save(path, writer='imagemagick', fps=30)
    
if __name__ == '__main__':
    optimizer = sys.argv[1]
    record_gif(optimizer)
