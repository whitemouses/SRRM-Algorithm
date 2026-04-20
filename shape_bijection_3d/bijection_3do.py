import time
import numpy as np
import open3d as o3d
import base
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# =========================
# 配置
# =========================
# MU_FILE = r"E:\HCP-main\HCP-main\dragon_vrip.pts"
# NU_FILE = r"E:\HCP-main\HCP-main\bun_zipper.pts"
MU_FILE = r"E:\HCP-main\HCP-main\bun_zipper.pts"
NU_FILE = r"E:\HCP-main\HCP-main\dragon_vrip.pts"

SAVE_EVERY = 20   # 每多少次迭代保存一次
SAVE_DIR = r"E:\HCP-main\HCP-main\outputs\screens"  # 截图保存目录（自己改）
SAVE_PREFIX = "dragon_to_bunny"  # 文件名前缀

PLOT_RMSE = True  # 最后是否画 rmse 曲线


NB_TREES = 16
SEED = 0

FORCE_N = 30000           # 匹配用点数
PLOT_MAX_POINTS = 80000   # 仅显示下采样

# 迭代参数：核心在 ALPHA
ITER_ITERS = 300          # 迭代轮数（想“无限”就设很大）
ALPHA = 0.15              # 每轮走多大步（0.05~0.3常用）
TOTAL_SECONDS = 30.0      # 让这 ITER_ITERS 轮大约播放 30 秒

SHOW_TARGET_B = True
POINT_SIZE = 2.0
BACKGROUND = np.asarray([1.0, 1.0, 1.0])  # 白底
POINT_SIZE_X = 3.0                        # 当前点更大
POINT_SIZE_Y = 1.5                        # 目标点更小（用同一个 point_size 也行，但这里用“交替显示法”，见下）
ERROR_COLOR = True                        # 是否按误差给 X 上色（推荐 True）


# =========================
# utils
# =========================
def read_pts(path):
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr
def colors_solid(P, rgb):
    return np.tile(np.array(rgb, dtype=np.float64)[None, :], (P.shape[0], 1))
def subsample_no_replace(X, N, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=N, replace=False)
    return X[idx]

def normalize_unit_ball(X, eps=1e-12):
    Xc = X - X.mean(axis=0, keepdims=True)
    mx = np.linalg.norm(Xc, axis=1).max()
    return Xc if mx < eps else Xc / mx

def downsample_for_plot(X, max_points, seed):
    if max_points is None or X.shape[0] <= max_points:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_points, replace=False)
    return X[idx]

def colors_by_height(P):
    z = P[:, 2]
    t = (z - z.min()) / (z.max() - z.min() + 1e-12)
    c = np.stack([t, 0.5 * (1.0 - t) + 0.2, 1.0 - t], axis=1)
    return np.clip(c, 0.0, 1.0)
def colors_solid(P, rgb):
    return np.tile(np.array(rgb, dtype=np.float64)[None, :], (P.shape[0], 1))

def colors_by_error(P, Q, eps=1e-12):
    """
    P: 当前点 (N,3)
    Q: 对应目标点 (N,3) 例如 Ymatch
    返回颜色：误差越大越红，误差越小越蓝/绿（非常直观）
    """
    d = np.linalg.norm(P - Q, axis=1)
    d0, d1 = float(d.min()), float(d.max())
    t = (d - d0) / (d1 - d0 + eps)

    # 蓝(好) -> 红(差)
    c = np.stack([t, 0.2 * (1.0 - t) + 0.2, 1.0 - t], axis=1)
    return np.clip(c, 0.0, 1.0)
def make_pcd(P, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


# =========================
# data
# =========================
A0 = read_pts(MU_FILE)
B0 = read_pts(NU_FILE)
if A0.shape[1] != 3 or B0.shape[1] != 3:
    raise RuntimeError("Expect 3D point clouds (3 columns).")

NA, NB = A0.shape[0], B0.shape[0]
N = min(FORCE_N, NA, NB)
A = A0 if NA == N else subsample_no_replace(A0, N, SEED)
B = B0 if NB == N else subsample_no_replace(B0, N, SEED + 1)

def normalize_01_isotropic(A, B, eps=1e-12):
    Z = np.vstack([A, B])
    mn = float(Z.min())
    mx = float(Z.max())
    s = max(mx - mn, eps)
    return (A - mn) / s, (B - mn) / s

A, B = normalize_01_isotropic(A, B)

# 显示用下采样（重要：显示用的点集必须与更新保持一致，所以我们直接在显示集上迭代）
X = downsample_for_plot(A, PLOT_MAX_POINTS, SEED + 10).copy()
Y = downsample_for_plot(B, PLOT_MAX_POINTS, SEED + 11).copy()

# 确保迭代匹配两边点数相等
n_vis = min(X.shape[0], Y.shape[0])
X = X[:n_vis]
Y = Y[:n_vis]

print(f"[Setup] match points = {n_vis}, ALPHA={ALPHA}, iters={ITER_ITERS}")

# =========================
# Open3D window with pause
# =========================
paused = {"v": False}
should_close = {"v": False}

def toggle_pause(vis):
    paused["v"] = not paused["v"]
    print("[Paused]" if paused["v"] else "[Resume]")
    return False

def request_close(vis):
    should_close["v"] = True
    print("[Quit requested]")
    return False

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("Iterative rematching (Open3D)", 1280, 720)
vis.register_key_callback(ord(" "), toggle_pause)
vis.register_key_callback(256, request_close)  # ESC


pcd_X = make_pcd(X, colors_solid(X, [1.0, 0.0, 0.0]))  # X 红色
vis.add_geometry(pcd_X)

if SHOW_TARGET_B:
    pcd_Y = make_pcd(Y, colors_solid(Y, [0.0, 0.0, 1.0]))  # Y 蓝色
    vis.add_geometry(pcd_Y)


opt = vis.get_render_option()
opt.background_color = BACKGROUND
opt.point_size = float(POINT_SIZE)

vis.poll_events()
vis.update_renderer()

os.makedirs(SAVE_DIR, exist_ok=True)

def save_screenshot(iter_id: int):
    # 先刷新一下，保证截图是最新一帧
    vis.poll_events()
    vis.update_renderer()
    out_path = os.path.join(SAVE_DIR, f"{SAVE_PREFIX}_iter{iter_id:04d}.png")
    vis.capture_screen_image(out_path, do_render=True)
    print(f"[Saved] screenshot -> {out_path}")

# 每轮 sleep，使总时长约为 TOTAL_SECONDS
sleep_per_iter = TOTAL_SECONDS / max(ITER_ITERS, 1)
print(f"[Timing] sleep_per_iter≈{sleep_per_iter:.4f}s, target_total≈{TOTAL_SECONDS:.1f}s")
print("[Run] SPACE pause/resume, ESC quit. End stays open.")

# =========================
# iterative rematching
# =========================
rmse_hist = []
iter_hist = []
for it in range(ITER_ITERS):
    if should_close["v"]:
        break

    while paused["v"] and not should_close["v"]:
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.03)


    # plan, _ = base.computeBijectiveBSPOT(
    #     np.ascontiguousarray(X.astype(np.float64)),
    #     np.ascontiguousarray(Y.astype(np.float64)),
    #     int(NB_TREES)
    # )



    num_rounds1 = 10 if it > 150 else 0
    plan = base.iter_match_dpq(
        np.ascontiguousarray(X.astype(np.float64)), np.ascontiguousarray(Y.astype(np.float64)),
        num_rounds=num_rounds1,
        z_mode="uniform01",
        verbose=False,
        return_history=False,
        z_count=2,
        p=int(NB_TREES),
        cycle=True,
        seed0=0,
        finalize=True
    )


    plan = np.asarray(plan, dtype=np.int64).reshape(-1)

    Ymatch = Y[plan]

    # 2) move X a small step toward matched Y
    X = (1.0 - ALPHA) * X + ALPHA * Ymatch

    # 3) show
    pcd_X.points = o3d.utility.Vector3dVector(X)
    pcd_X.colors = o3d.utility.Vector3dVector(colors_solid(X, [1.0, 0.0, 0.0]))  # 始终红色
    vis.update_geometry(pcd_X)
    vis.poll_events()
    vis.update_renderer()

    # 计算并记录 rmse（每轮都记录，最后画曲线用）
    rmse = float(np.sqrt(np.mean(np.sum((X - Ymatch) ** 2, axis=1))))
    rmse_hist.append(rmse)
    iter_hist.append(it)

    # 每 10 次打印一次
    if (it % 10) == 0:
        print(f"[Iter {it:04d}] rmse={rmse:.6g}")

    # 每 SAVE_EVERY 次保存一次截图（也可以包含最后一次）
    if it < 50:
        save_now = (it % 5 == 0)
    else:
        save_now = (it % 50 == 0)

    if save_now or (it == ITER_ITERS - 1):
        save_screenshot(it)

    time.sleep(sleep_per_iter)


if PLOT_RMSE and len(rmse_hist) > 0:
    plt.figure()
    plt.plot(iter_hist, rmse_hist)
    plt.xlabel("iteration")
    plt.ylabel("RMSE(X vs matched Y)")
    plt.title("RMSE over iterations")
    plt.grid(True)
    # 可选：保存 rmse 曲线
    rmse_fig = os.path.join(SAVE_DIR, f"{SAVE_PREFIX}_rmse.png")
    plt.savefig(rmse_fig, dpi=200, bbox_inches="tight")
    print(f"[Saved] rmse plot -> {rmse_fig}")
    plt.show()
# keep window open


print("[Done] Iteration finished. Window stays open. ESC or close window to quit.")
while not should_close["v"]:
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.03)

vis.destroy_window()
