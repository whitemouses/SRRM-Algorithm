# %%

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as pl
import sys

sys.path.append("../src")
from utility import HCP, base
import time
import ot
import torch
import ot

import os
from pathlib import Path

# %%

r = np.random.RandomState(42)


def im2mat(img):
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))


def mat2im(X, shape):
    return X.reshape(shape)


def minmax(img):
    return np.clip(img, 0, 1)


# %% md

# Green_Forest color transfer

# %%

I2 = pl.imread('figure/ocean_day.jpg').astype(np.float64) / 256
I1 = pl.imread('figure/ocean_sunset.jpg').astype(np.float64) / 256

X1 = im2mat(I1)
X2 = im2mat(I2)
ori_dat = X1
des_dat = X2

N = ori_dat.shape[0]
a, b = np.ones((N,)) / N, np.ones((N,)) / N


# %%

t0 = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(42)

x1_torch = torch.tensor(ori_dat).to(device=device).requires_grad_(True)
s_des_dat = des_dat[np.random.choice(des_dat.shape[0], ori_dat.shape[0]), :]
x2_torch = torch.tensor(s_des_dat).to(device=device)

x2o = base.hilbert_order(s_des_dat)

lr = 2e4
nb_iter_max = 30
x_all_bsp_top = np.zeros((nb_iter_max, ori_dat.shape[0], 3), dtype=np.float32)

for i in range(nb_iter_max):
    # --- compute matching plan using current x1 (NON-diff, same as your hilbert_order usage) ---
    x1n = x1_torch.cpu().detach().numpy()
    x1o = base.hilbert_order(x1n)

    loss = torch.mean(torch.pow((x1_torch[x1o, :] - x2_torch[x2o, :]), 2)) * 3
    loss = torch.sqrt(loss)
    loss.backward()

    # --- projected GD step (your original style) ---
    with torch.no_grad():
        grad = x1_torch.grad
        x1_torch -= grad * lr / (1.0 + i / 5e1)
        x1_torch.grad.zero_()
        x_all_bsp_top[i, :, :] = x1_torch.detach().cpu().numpy()

xb = x1_torch.detach().cpu().numpy()
Image_hcp = minmax(mat2im(xb, I1.shape))
t_bsp = time.time() - t0
print('Time is ', t_bsp, ' seconds.')



# %% md

# Forest_road color transfer


# %%

t0 = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(42)

x1_torch = torch.tensor(des_dat).to(device=device).requires_grad_(True)
s_ori_dat = ori_dat[np.random.choice(ori_dat.shape[0], des_dat.shape[0]), :]
x2_torch = torch.tensor(s_ori_dat).to(device=device)

x2o = base.hilbert_order(s_ori_dat)
lr = 2e4
nb_iter_max = 30
x_all_bsp_top2 = np.zeros(((nb_iter_max, des_dat.shape[0], 3)), dtype=np.float32)

for i in range(nb_iter_max):
    x1n = x1_torch.cpu().detach().numpy()
    x1o = base.hilbert_order(x1n)

    loss = torch.mean(torch.pow((x1_torch[x1o, :] - x2_torch[x2o, :]), 2)) * 3
    loss = torch.sqrt(loss)
    loss.backward()

    # --- projected GD step (your original style) ---
    with torch.no_grad():
        grad = x1_torch.grad
        x1_torch -= grad * lr / (1.0 + i / 5e1)
        x1_torch.grad.zero_()
        x_all_bsp_top2[i, :, :] = x1_torch.detach().cpu().numpy()

xb = x1_torch.detach().cpu().numpy()
Image_bsp2 = minmax(mat2im(xb, I2.shape))
t_bsp = time.time() - t0
print('Time is ', t_bsp, ' seconds.')

# %%

iterations = [1, 5, 10, 15, 20, 30]
ii = 0
pl.figure(1, figsize=(20, 6))

for i in iterations:
    ii += 1
    pl.subplot(1, 6, ii)
    Image_i = minmax(mat2im(x_all_bsp_top[i - 1], I1.shape))
    pl.imshow(Image_i)
    # pl.axis('off')
    pl.title('Iteration-' + str(i), fontsize=20)
    if ii == 1:
        pl.ylabel('HCP', fontsize=20)
        pl.xticks([])
        pl.yticks([])
    else:
        pl.axis('off')

pl.tight_layout()
pl.savefig('Results/color_hcp.png')
pl.show()

# %%

iterations = [1, 5, 10, 15, 20, 30]
ii = 0
pl.figure(1, figsize=(20, 6))

for i in iterations:
    ii += 1
    pl.subplot(1, 6, ii)
    Image_i = minmax(mat2im(x_all_bsp_top2[i - 1], I2.shape))
    pl.imshow(Image_i)
    # pl.axis('off')
    pl.title('Iteration-' + str(i), fontsize=20)
    if ii == 1:
        pl.ylabel('HCP', fontsize=20)
        pl.xticks([])
        pl.yticks([])
    else:
        pl.axis('off')

pl.tight_layout()
pl.savefig('Results/color1_hcp.png')
pl.show()
