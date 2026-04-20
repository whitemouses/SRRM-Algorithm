# %%

import sys

sys.path.append('./gsw/')

import numpy as np
from gsw.gsw import GSW
from gsw.gsw_utils import w2, load_data

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch import optim

from tqdm import tqdm
from IPython import display
import time
import pickle
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random
import os

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# %%

np.random.seed(10)


def load_data1(name='25gaussians', n_samples=1000, rng=None):
    rng = np.random.default_rng(1) if rng is None else rng
    if name == '25gaussians':
        centers = np.linspace(-2.0, 2.0, 5)
        C = np.array([(cx, cy) for cx in centers for cy in centers])
        idx = rng.integers(0, len(C), size=n_samples)
        X = C[idx] + 0.02 * rng.standard_normal(size=(n_samples, 2))
    elif name == 'circle':
        th = rng.uniform(0, 2*np.pi, size=n_samples)
        r  = 2 + 0.02*rng.standard_normal(size=n_samples)
        X = np.stack([r*np.cos(th), r*np.sin(th)], axis=1)
    elif name == 'swiss_roll':
        t = rng.uniform(3*np.pi/2, 9*np.pi/2, size=n_samples)
        x = t * np.cos(t); y = t * np.sin(t)
        X = np.stack([x, y], axis=1)
        X = (X - X.min(0)) / (X.ptp(0) + 1e-12) * 4 - 2
    elif name == 'gussianx':
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, size=(n_samples, 2))
    elif name == 'puma':
        np_nu1t = np.load("puma.npy")
        X = np_nu1t
    elif name == 'biankuang':
        np_nu1t = np.load("biankuang.npy")
        X =  np_nu1t
    else:
        raise ValueError("name must be '25gaussians' | 'circle' | 'swiss_roll'")
    X = torch.from_numpy(X).float()
    return X




# choose a dataset and show it
dataset_name = 'swiss_roll'  # 'circle' or 'swiss_roll'  '25gaussians'

N = 1000
X = load_data1(name=dataset_name, n_samples=N)
eps = 1e-12
xmin = X.min()
xmax = X.max()
X = (X - xmin) / (xmax - xmin + eps)
# X -= X.mean(dim=0)[np.newaxis, :]
meanX = [0.5,0.5]

_, d = X.shape
# fig = plt.figure(figsize=(5, 5))
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

# %%

# save results
results_folder = './saved_results_flows'
if not os.path.isdir(results_folder):
    os.mkdir(results_folder)

foldername = os.path.join(results_folder, 'figures')
if not os.path.isdir(foldername):
    os.mkdir(foldername)

foldername = os.path.join(results_folder, 'figures', dataset_name + '_Comparison')
if not os.path.isdir(foldername):
    os.mkdir(foldername)

# %%

nofiterations = 201

w2_dist = np.nan * np.zeros((nofiterations, 4))
maxw2_dist = np.nan * np.zeros((nofiterations, 3))

titles = [
    'SW(Linear)',
    'GSW(Poly 5)',          # 实际含义：linear + poly3 + poly5
    'Max-SW(Max Linear)',
    'Max-GSW(Max Poly 5)',  # 实际含义：max linear + max poly3 + max poly5
    'HCP',
    'BSP',
    'SRRM'
]

# Define the initial distribution
temp = np.random.normal(loc=meanX, scale=.25, size=(X.shape[0], X.shape[1]))

# Define the optimizers
Y = []
optimizer = []
gsw_groups = []

# 7个待优化分布
for k in range(7):
    Y.append(torch.tensor(temp, dtype=torch.float, device=device, requires_grad=True))
    optimizer.append(optim.Adam([Y[k]], lr=1e-2))

# 前4个方法，按“组”来定义
# 0: SW(Linear)                  = linear
# 1: GSW(Poly 5)                = linear + poly3 + poly5
# 2: Max-SW(Max Linear)         = linear
# 3: Max-GSW(Max Poly 5)        = linear + poly3 + poly5
gsw_groups.append([
    GSW(ftype='linear', degree=1, nofprojections=10)
])

gsw_groups.append([
    GSW(ftype='linear', degree=1, nofprojections=10),
    GSW(ftype='poly', degree=3, nofprojections=10),
    GSW(ftype='poly', degree=5, nofprojections=10),
])

gsw_groups.append([
    GSW(ftype='linear', degree=1, nofprojections=10)
])

gsw_groups.append([
    GSW(ftype='linear', degree=1, nofprojections=10),
    GSW(ftype='poly', degree=3, nofprojections=10),
    GSW(ftype='poly', degree=5, nofprojections=10),
])

fig = plt.figure(figsize=(45, 4), constrained_layout=True)
grid = fig.add_gridspec(
    1, 8,
    width_ratios=[3, 1, 1, 1, 1, 1, 1, 1],
    wspace=0.2
)

losspp = np.ones(nofiterations) * 2

for i in range(nofiterations):
    # -------- 0,1: 普通 SW / GSW --------
    for k in range(2):
        loss_ = 0
        for g in gsw_groups[k]:
            loss_ += g.gsw(X.to(device), Y[k].to(device))

        optimizer[k].zero_grad()
        loss_.backward()
        optimizer[k].step()

        w2_dist[i, k] = w2(X.detach().cpu().numpy(), Y[k].detach().cpu().numpy())

    # -------- 2,3: Max-SW / Max-GSW --------
    for k in range(2, 4):
        loss_ = 0
        for g in gsw_groups[k]:
            loss_ += g.max_gsw(X.to(device), Y[k].to(device), iterations=50, lr=1e-2)

        optimizer[k].zero_grad()
        loss_.backward()
        optimizer[k].step()

        w2_dist[i, k] = w2(X.detach().cpu().numpy(), Y[k].detach().cpu().numpy())

    # -------- 4,5,6: HCP / BSP / SRRM --------
    for k in range(4, 7):
        if k == 4:
            g = GSW(ftype='HCP', degree=1, nofprojections=1, lossp=losspp[i])
        elif k == 5:
            g = GSW(ftype='BSP', degree=1, nofprojections=1, lossp=losspp[i])
        else:
            g = GSW(ftype='SRRM', degree=1, nofprojections=1, lossp=losspp[i])

        loss_ = g.gsw(X.to(device), Y[k].to(device))

        optimizer[k].zero_grad()
        loss_.backward()
        optimizer[k].step()

        maxw2_dist[i, k - 4] = w2(X.detach().cpu().numpy(), Y[k].detach().cpu().numpy())

    # -------- 绘图 --------
    if (i % 10 == 0) or (i == nofiterations - 1):
        for k in range(4):
            temp_plot = Y[k].detach().cpu().numpy()
            plt.subplot(grid[0, k + 1])
            plt.cla()
            plt.scatter(X[:, 0], X[:, 1], c='b', s=5)
            plt.scatter(temp_plot[:, 0], temp_plot[:, 1], c='r', s=5)
            plt.title(titles[k], fontsize=22)

        for k in range(4, 7):
            temp_plot = Y[k].detach().cpu().numpy()
            plt.subplot(grid[0, k + 1])
            plt.cla()
            plt.scatter(X[:, 0], X[:, 1], c='b', s=5)
            plt.scatter(temp_plot[:, 0], temp_plot[:, 1], c='r', s=5)
            plt.title(titles[k], fontsize=22)

        plt.subplot(grid[0, 0])
        plt.cla()
        plt.plot(np.log10(w2_dist[:, 0]), linewidth=3, c='orange', label='SW(Linear)')
        plt.plot(np.log10(w2_dist[:, 1]), linewidth=3, c='violet', label='GSW(Poly5)')
        plt.plot(np.log10(w2_dist[:, 2]), linewidth=3, c='darkorange', linestyle='dashed', label='Max-SW(Linear)')
        plt.plot(np.log10(w2_dist[:, 3]), linewidth=3, c='darkviolet', linestyle='dashed', label='Max-GSW(Poly5)')
        plt.plot(np.log10(maxw2_dist[:, 0]), linewidth=3, c='y', label='HCP')
        plt.plot(np.log10(maxw2_dist[:, 1]), linewidth=3, c='blue', label='BSP')
        plt.plot(np.log10(maxw2_dist[:, 2]), linewidth=3, c='red', label='SRRM')

        plt.title('2-Wasserstein Distance', fontsize=22)
        plt.ylabel(r'$Log_{10}(W_2)$', fontsize=22)
        plt.legend(fontsize=15, loc='lower left')

        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(1e-5)
        plt.tight_layout()
        fig.savefig(foldername + '/img%03d.png' % i, dpi=200, bbox_inches='tight')






