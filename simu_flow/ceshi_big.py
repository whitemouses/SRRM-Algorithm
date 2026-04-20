
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
    elif name == 'Y3':
        np_nu1t = np.load("Y3_points.npy")
        X =  np_nu1t
    elif name == 'Y4':
        np_nu1t = np.load("Y4_points.npy")
        X =  np_nu1t
    else:
        raise ValueError("name must be '25gaussians' | 'circle' | 'swiss_roll'")
    X = torch.from_numpy(X).float()
    return X




# choose a dataset and show it
dataset_name = 'Y4'  # 'Y3' or 'Y4'

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







# %% ---------------------------
# some settings (只保留 HCP / BSP / TCP)
# ---------------------------
nofiterations = 201

titles = ['HCP', 'BSP', 'SRRM']
ftypes  = ['HCP', 'BSP', 'SRRM']
n_methods = 3

w2_dist = np.full((nofiterations, n_methods), np.nan, dtype=np.float64)

time_loss = np.zeros((nofiterations, n_methods), dtype=np.float64)
time_cum  = np.zeros((nofiterations, n_methods), dtype=np.float64)

def sync_if_cuda(dev):
    if hasattr(dev, "type") and dev.type == "cuda":
        torch.cuda.synchronize()

# %% ---------------------------
# Define the initial distribution (初始化 Y)
# ---------------------------
temp = np.random.normal(loc=meanX, scale=.25, size=(X.shape[0], X.shape[1]))

Y = []
optimizer = []

for k in range(n_methods):
    Yk = torch.tensor(temp, dtype=torch.float, device=device, requires_grad=True)
    Y.append(Yk)
    optimizer.append(optim.Adam([Yk], lr=1e-2))

# 你的 lossp 调度（保持不变）
losspp = np.ones(nofiterations) * 2

# %% ---------------------------
# Figure layout: 1行5列
# [大图W2] [大图Time] [HCP散点] [BSP散点] [TCP散点]
# ---------------------------
fig = plt.figure(figsize=(25, 4), constrained_layout=True)
grid = fig.add_gridspec(1, 4, width_ratios=[3, 1, 1, 1], wspace=0.25)
ax_w2 = fig.add_subplot(grid[0, 0])
ax_scatter = [fig.add_subplot(grid[0, 1 + k]) for k in range(3)]

for i in range(nofiterations):

    for k in range(n_methods):
        # 每次迭代重新构造 GSW（你原来就是这么做的）
        if ftypes[k] == 'HCP':
            g = GSW(ftype='HCP', degree=1, nofprojections=1, lossp=losspp[i])
        elif ftypes[k] == 'BSP':
            g = GSW(ftype='BSP', degree=1, nofprojections=1, lossp=losspp[i])
        else:
            g = GSW(ftype='SRRM', degree=1, nofprojections=1, lossp=losspp[i])

        # ---------- 只计 "loss forward" 时间 ----------
        sync_if_cuda(device)
        t0 = time.perf_counter()
        loss_ = g.gsw(X.to(device), Y[k].to(device))
        sync_if_cuda(device)
        t1 = time.perf_counter()

        time_loss[i, k] = (t1 - t0)

        # ---------- 训练更新（不计时） ----------
        optimizer[k].zero_grad()
        loss_.backward()
        optimizer[k].step()

        # W2 评估（不计时 or 你可以单独再计一个评估时间）
        w2_dist[i, k] = w2(X.detach().cpu().numpy(), Y[k].detach().cpu().numpy())

    # ---------- 累计时间 ----------
    if i == 0:
        time_cum[i, :] = time_loss[i, :]
    else:
        time_cum[i, :] = time_cum[i-1, :] + time_loss[i, :]

    # ---------------------------
    # Save / display figure every 10 iters
    # ---------------------------
    if (i % 10 == 0) or (i == nofiterations - 1):

        # --- 右侧散点 ---
        for k in range(3):
            ax = ax_scatter[k]
            ax.cla()
            tempk = Y[k].detach().cpu().numpy()
            ax.scatter(X[:, 0], X[:, 1], c='b', s=5)
            ax.scatter(tempk[:, 0], tempk[:, 1], c='r', s=5)
            ax.set_title(titles[k], fontsize=18)
            ax.set_xticks([])
            ax.set_yticks([])

        # --- 左1：W2 ---
        ax_w2.cla()
        ax_w2.plot(np.log10(w2_dist[:, 0]), linewidth=3, c='y',    label='HCP')
        ax_w2.plot(np.log10(w2_dist[:, 1]), linewidth=3, c='blue', label='BSP')
        ax_w2.plot(np.log10(w2_dist[:, 2]), linewidth=3, c='red',  label='SRRM')
        ax_w2.set_title('2-Wasserstein Distance', fontsize=18)
        ax_w2.set_ylabel(r'$Log_{10}(W_2)$', fontsize=16)
        ax_w2.legend(fontsize=12, loc='lower left')

        # --- 打印时间（只算loss forward） ---
        win = 10
        lo = max(0, i - win + 1)

        cur = time_loss[i, :]  # 当前epoch的loss forward耗时
        cum = time_cum[i, :]  # 累计到当前
        avg = cum / float(i + 1)  # 从0到当前的“每epoch平均耗时”

        print(
            f"[iter {i:03d}] "
            f"cur(s): HCP={cur[0]:.6f}, BSP={cur[1]:.6f}, TCP={cur[2]:.6f} | "
            f"avg_epoch(s): HCP={avg[0]:.6f}, BSP={avg[1]:.6f}, TCP={avg[2]:.6f} | "
            f"cum(s): HCP={cum[0]:.2f}, BSP={cum[1]:.2f}, TCP={cum[2]:.2f}"
        )


        display.clear_output(wait=True)
        display.display(fig)
        fig.savefig(foldername + f'/img{i:03d}.png', dpi=200, bbox_inches='tight')