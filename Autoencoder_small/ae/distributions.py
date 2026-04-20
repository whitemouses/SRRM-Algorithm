import numpy as np

import torch
from sklearn.datasets import make_circles
from PIL import Image
#
def rand_cirlce2d1( batch_size, device="cpu"):

    # 1. 读图片并转成灰度
    image_path = 'E:\HCP-main\HCP-main\Autoencoder_small/ae/4.png'
    img = Image.open(image_path).convert("L")  # L: 8bit 灰度
    arr = np.array(img, dtype=np.float32) / 255.0  # 归一化到 [0,1]

    # 2. 用阈值选出“前景像素”（图案部分）
    mask = arr < 0.5
    ys, xs = np.where(mask)  # 行、列索引

    if len(xs) == 0:
        raise ValueError("阈值太高，整张图都被当成背景了，没有可采样的点。")

    # 3. 像素坐标归一化到 [-1,1]×[-1,1]
    H, W = arr.shape
    xs_norm = (xs.astype(np.float32) / (W - 1))
    ys_norm = (ys.astype(np.float32) / (H - 1))


    points = np.stack([xs_norm, ys_norm], axis=1)  # (N_points, 2)

    # 4. 从这些点里随机抽 batch_size 个
    idx = np.random.randint(0, points.shape[0], size=batch_size)
    pts = points[idx].copy()

    return torch.from_numpy(pts).to(device).float()





#
def rand_cirlce2d( batch_size, device="cpu"):
    # 1. 读图片并转成灰度
    image_path = 'E:\HCP-main\HCP-main\Autoencoder_small/ae/13.png'
    img = Image.open(image_path).convert("L")  # L: 8bit 灰度
    arr = np.array(img, dtype=np.float32) / 255.0  # 归一化到 [0,1]

    # 2. 用阈值选出“前景像素”（图案部分）
    mask = arr < 0.5
    ys, xs = np.where(mask)  # 行、列索引

    if len(xs) == 0:
        raise ValueError("阈值太高，整张图都被当成背景了，没有可采样的点。")

    # 3. 像素坐标归一化到 [-1,1]×[-1,1]
    H, W = arr.shape
    xs_norm = (xs.astype(np.float32) / (W - 1))
    ys_norm = (ys.astype(np.float32) / (H - 1))

    points = np.stack([xs_norm, ys_norm], axis=1)  # (N_points, 2)

    # 4. 从这些点里随机抽 batch_size 个
    idx = np.random.randint(0, points.shape[0], size=batch_size)
    pts = points[idx].copy()

    return torch.from_numpy(pts).to(device).float()




def rand_uniform2d(batch_size):
    z = np.random.uniform(low=0.1, high=0.9, size=(batch_size, 2))
    return torch.from_numpy(z).float()


def rand(dim_size):
    def _rand(batch_size):
        return torch.rand((batch_size, dim_size))
    return _rand


def randn(dim_size):
    def _randn(batch_size):
        return torch.randn((batch_size, dim_size))
    return _randn


#
# if __name__ == "__main__":
#     device = "cpu"
#     pts = rand_cirlce2d(500, device=device)
#
#     pts_np = pts.cpu().numpy()
#
#     plt.figure(figsize=(5, 5))
#     # 如果想让图像方向跟原图一样，可以把 y 轴取反：-pts_np[:,1]
#     plt.scatter(pts_np[:, 0], -pts_np[:, 1], s=3)
#     plt.xlim(-1, 1)
#     plt.ylim(-1, 1)
#     plt.gca().set_aspect('equal', adjustable='box')  # 保持比例 1:1
#     plt.title("Sampled points from image")
#     plt.show()