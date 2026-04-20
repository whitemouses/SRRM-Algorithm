import os
import numpy as np
import open3d as o3d
from typing import Optional

# ====== 改这里：你的 .ply 路径 ======
ply_path = r"E:\HCP-main\HCP-main\dragon_vrip.ply"

# ====== 输出 .pts 保存路径（空=同目录同名）======
out_pts_path = ""  # 例如 r"E:\out\dragon_recon.pts"

# ====== 如果 PLY 是网格(mesh)，是否采样固定点数 ======
sample_points: Optional[int] = 8000  # 或 None

# 采样方式： "poisson"（更均匀）或 "uniform"
sample_method = "poisson"


def to_points_from_ply(path: str, n_samples: Optional[int]) -> np.ndarray:
    # 先按点云读
    pcd = o3d.io.read_point_cloud(path)
    if pcd.has_points() and len(pcd.points) > 0:
        X = np.asarray(pcd.points, dtype=np.float64)
        if n_samples is not None and X.shape[0] > n_samples:
            idx = np.random.choice(X.shape[0], size=n_samples, replace=False)
            X = X[idx]
        return X

    # 再按网格读
    mesh = o3d.io.read_triangle_mesh(path)
    if not mesh.has_triangles() or len(mesh.triangles) == 0:
        raise ValueError("Cannot read as point cloud or mesh: {}".format(path))

    if n_samples is None:
        return np.asarray(mesh.vertices, dtype=np.float64)

    mesh.compute_vertex_normals()
    if sample_method.lower() == "poisson":
        pcd2 = mesh.sample_points_poisson_disk(number_of_points=int(n_samples))
    else:
        pcd2 = mesh.sample_points_uniformly(number_of_points=int(n_samples))
    return np.asarray(pcd2.points, dtype=np.float64)


def save_pts(path: str, X: np.ndarray) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savetxt(path, X, fmt="%.9g")


def show_open3d_and_save(
    X: np.ndarray,
    title: str = "point cloud",
    point_size: int = 2,
    screenshot_path: str = "",
    background_white: bool = True,
):
    """
    打开 Open3D 窗口显示点云，并自动保存一张截图（如果 screenshot_path 非空）。
    保存截图后窗口会保持打开，让你继续旋转查看；关闭窗口后程序结束。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X.astype(np.float64))

    # 按坐标归一化上色（更立体）
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    colors = (X - mn) / (mx - mn + 1e-12)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=720)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    if background_white:
        opt.background_color = np.asarray([1.0, 1.0, 1.0])
    else:
        opt.background_color = np.asarray([0.0, 0.0, 0.0])

    # 先渲染一帧，保证截图不是黑屏
    vis.poll_events()
    vis.update_renderer()

    # 自动保存截图
    if screenshot_path:
        out_dir = os.path.dirname(screenshot_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        vis.capture_screen_image(screenshot_path, do_render=True)
        print("[OK] Screenshot saved:", screenshot_path)

    # 继续显示窗口（你可以旋转缩放），关掉窗口才结束
    vis.run()
    vis.destroy_window()


# ====== 顺着执行：右键运行即可 ======
if not os.path.isfile(ply_path):
    raise FileNotFoundError(ply_path)

X = to_points_from_ply(ply_path, sample_points)

# 输出路径
if not out_pts_path:
    out_pts_path = os.path.splitext(ply_path)[0] + ".pts"

save_pts(out_pts_path, X)

print("[OK] Loaded:", ply_path)
print("     Points:", X.shape[0], "Dim:", X.shape[1])
print("     Saved :", out_pts_path)

# 截图保存路径：默认和 .pts 同目录同名 .png
screenshot_path = os.path.splitext(out_pts_path)[0] + ".png"

show_open3d_and_save(
    X,
    title=os.path.basename(out_pts_path),
    point_size=3,                 # 想更粗就调大
    screenshot_path=screenshot_path,
    background_white=True
)