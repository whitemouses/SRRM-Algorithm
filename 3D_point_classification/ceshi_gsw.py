import os
import numpy as np
import random

from pathlib import Path
import plotly.graph_objects as go

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import time
import sys
import torch
from torch import optim
sys.path.append("../src")
from utility import HCP

def Maxsw(x, y, nproj=500):
    proj = np.zeros((nproj, 2))

    for i in range(nproj):
        proj[i, 0] = np.cos(i * np.pi * 2 / nproj)
        proj[i, 1] = np.sin(i * np.pi * 2 / nproj)
    xp = x @ proj.T
    yp = y @ proj.T

    xy = np.mean((np.sort(xp, 0) - np.sort(yp, 0)) ** 2, 0)
    return np.sqrt(np.max(xy))


# GSW
class GSW():
    def __init__(self, ftype='linear', nofprojections=10, degree=2, use_cuda=False):
        self.ftype = ftype
        self.nofprojections = nofprojections
        self.degree = degree
        if torch.cuda.is_available() and use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def gsw(self, X, Y, theta=None):

        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm and M == N
        if theta is None:
            theta = self.random_slice(dn)

        Xslices = self.get_slice(X, theta)
        Yslices = self.get_slice(Y, theta)

        Xslices_sorted = torch.sort(Xslices, dim=0)[0]
        Yslices_sorted = torch.sort(Yslices, dim=0)[0]
        return torch.sqrt(torch.mean((Xslices_sorted - Yslices_sorted) ** 2))

    def get_slice(self, X, theta):
        if self.ftype == 'linear':
            return self.linear(X, theta)
        elif self.ftype == 'poly':
            return self.poly(X, theta)
        else:
            raise Exception('Defining function not implemented')

    def random_slice(self, dim):
        if self.ftype == 'linear':
            theta = torch.randn((self.nofprojections, dim))
            theta = torch.stack([th / torch.sqrt((th ** 2).sum()) for th in theta])
        elif self.ftype == 'poly':
            dpoly = self.homopoly(dim, self.degree)
            theta = torch.randn((self.nofprojections, dpoly))
            theta = torch.stack([th / torch.sqrt((th ** 2).sum()) for th in theta])
        return theta.to(self.device)

    def linear(self, X, theta):
        if len(theta.shape) == 1:
            return torch.matmul(X, theta)
        else:
            return torch.matmul(X, theta.t())

    def poly(self, X, theta):
        N, d = X.shape
        assert theta.shape[1] == self.homopoly(d, self.degree)
        powers = list(self.get_powers(d, self.degree))
        HX = torch.ones((N, len(powers))).to(self.device)
        for k, power in enumerate(powers):
            for i, p in enumerate(power):
                HX[:, k] *= X[:, i] ** p
        if len(theta.shape) == 1:
            return torch.matmul(HX, theta)
        else:
            return torch.matmul(HX, theta.t())

    def get_powers(self, dim, degree):
        if dim == 1:
            yield (degree,)
        else:
            for value in range(degree + 1):
                for permutation in self.get_powers(dim - 1, degree - value):
                    yield (value,) + permutation

    def homopoly(self, dim, degree):
        return len(list(self.get_powers(dim, degree)))


def main():
    modes = ['linear', 'poly', 'poly']
    titles = ['Linear', 'Poly 3', 'Poly 5']
    degrees = [1, 3, 5]
    gsw = list()

    for k in range(3):
        gsw.append([GSW(ftype=modes[j], degree=degrees[j], nofprojections=10) for j in range(k + 1)])


    n_runs = 20

    acc_list = []
    time_train_train_list = []
    time_test_train_list = []
    time_knn_list = []
    time_total_list = []

    def read_off(file):
        if 'OFF' != file.readline().strip():
            raise ('Not a valid OFF header')
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        return verts, faces

    def visualize_rotate(data):
        x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
        frames = []

        def rotate_z(x, y, z, theta):
            w = x + 1j * y
            return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

        for t in np.arange(0, 10.26, 0.1):
            xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
            frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
        fig = go.Figure(data=data,
                        layout=go.Layout(
                            updatemenus=[dict(type='buttons',
                                              showactive=False,
                                              y=1,
                                              x=0.8,
                                              xanchor='left',
                                              yanchor='bottom',
                                              pad=dict(t=45, r=10),
                                              buttons=[dict(label='Play',
                                                            method='animate',
                                                            args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                             transition=dict(duration=0),
                                                                             fromcurrent=True,
                                                                             mode='immediate'
                                                                             )]
                                                            )
                                                       ])]
                        ),
                        frames=frames
                        )

        return fig

    def pcshow(xs, ys, zs):
        data = [go.Scatter3d(x=xs, y=ys, z=zs,
                             mode='markers')]
        fig = visualize_rotate(data)
        fig.update_traces(marker=dict(size=2,
                                      line=dict(width=2,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.show()

    # %%

    class PointSampler(object):
        def __init__(self, output_size):
            assert isinstance(output_size, int)
            self.output_size = output_size

        def triangle_area(self, pt1, pt2, pt3):
            side_a = np.linalg.norm(pt1 - pt2)
            side_b = np.linalg.norm(pt2 - pt3)
            side_c = np.linalg.norm(pt3 - pt1)
            s = 0.5 * (side_a + side_b + side_c)
            return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

        def sample_point(self, pt1, pt2, pt3):

            s, t = sorted([random.random(), random.random()])
            f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
            return (f(0), f(1), f(2))

        def __call__(self, mesh):
            verts, faces = mesh
            verts = np.array(verts)
            areas = np.zeros((len(faces)))

            for i in range(len(areas)):
                areas[i] = (self.triangle_area(verts[faces[i][0]],
                                               verts[faces[i][1]],
                                               verts[faces[i][2]]))

            sampled_faces = (random.choices(faces,
                                            weights=areas,
                                            cum_weights=None,
                                            k=self.output_size))

            sampled_points = np.zeros((self.output_size, 3))

            for i in range(len(sampled_faces)):
                sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                       verts[sampled_faces[i][1]],
                                                       verts[sampled_faces[i][2]]))

            return sampled_points

    class Normalize(object):
        def __call__(self, pointcloud):
            assert len(pointcloud.shape) == 2

            norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
            norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

            return norm_pointcloud

    # %%

    path = Path("E:\HCP-main\HCP-main/3D_point_classification/archive\ModelNet10")

    folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path / dir)]
    classes = {folder: i for i, folder in enumerate(folders)};

    print(classes)

    data = pd.read_csv('metadata_modelnet10.csv')
    data.columns = ['object_id', 'type', 'split', 'object_path']
    data.head(5)
    #
    # # %%
    #
    print(np.sum(data.split == 'train'))
    train_data = data[data['split'] == 'train'].reset_index(drop=True)
    print(np.sum(data.split == 'test'))
    test_data = data[data['split'] == 'test'].reset_index(drop=True)
    seed = 0


    for run in range(n_runs):
        print(f"\n========== Run {run + 1}/{n_runs} ==========")
        np.random.seed(seed + run)

        subclass = classes.keys()


        Ktrain = 60 * 10
        Ktest = 20 * 10

        train_3D = [None] * Ktrain
        train_label = np.zeros(Ktrain)
        i = 0
        j = 0

        KK = 2000

        for category in subclass:
            # 1) 先解决 type 的匹配：night_stand 这类在 csv 里叫 night
            if category == 'night_stand':
                type_name = 'night'
            else:
                type_name = category

            cat_df = train_data[train_data['type'] == type_name]
            # print("category:", category, "(type in csv:", type_name, ") train samples:", len(cat_df))
            n_per_class = min(int(Ktrain / 10), len(cat_df))


            temple_data = cat_df.sample(n=n_per_class, random_state=seed + run).reset_index(drop=True)

            for k in range(temple_data.shape[0]):
                temple_path = str(temple_data.object_path[k])

                # 2) 修正 object_path 里的 night -> night_stand
                if category == 'night_stand':
                    # 兼容 / 和 \ 两种写法，都替换一次
                    if temple_path.startswith('night/'):
                        temple_path = 'night_stand/' + temple_path[len('night/'):]
                    elif temple_path.startswith('night\\'):
                        temple_path = 'night_stand\\' + temple_path[len('night\\'):]

                with open(path / temple_path, 'r') as f:
                    verts, faces = read_off(f)
                    pointcloud = PointSampler(KK)((verts, faces))
                    train_3D[i] = Normalize()(pointcloud)
                    train_label[i] = j
                    i += 1

            # print("this class used:", k + 1)
            j += 1
            # print("total so far:", i)

        test_3D = [None] * Ktest
        test_label = np.zeros(Ktest)
        i = 0
        j = 0

        for category in subclass:
            if category == 'night_stand':
                type_name = 'night'
            else:
                type_name = category

            cat_df = test_data[test_data['type'] == type_name]
            # print("category:", category, "(type in csv:", type_name, ") test samples:", len(cat_df))

            n_per_class = min(int(Ktest / 10), len(cat_df))

            temple_data = cat_df.sample(n=n_per_class, random_state=seed + run).reset_index(drop=True)

            for k in range(temple_data.shape[0]):
                temple_path = str(temple_data.object_path[k])

                # 同样修正路径
                if category == 'night_stand':
                    if temple_path.startswith('night/'):
                        temple_path = 'night_stand/' + temple_path[len('night/'):]
                    elif temple_path.startswith('night\\'):
                        temple_path = 'night_stand\\' + temple_path[len('night\\'):]

                with open(path / temple_path, 'r') as f:
                    verts, faces = read_off(f)
                    pointcloud = PointSampler(KK)((verts, faces))
                    test_3D[i] = Normalize()(pointcloud)
                    test_label[i] = j
                    i += 1

            # print("this class used:", k + 1)
            j += 1
            # print("total so far:", i)

        # =========================#=========================#=========================#=========================#=========================

        t_total = time.time()

        # 如果 HCP 里或别的地方用到了随机性，可以根据 run 改 seed
        np.random.seed(seed + run)

        # ---------- 1) 训练–训练 HCP 距离矩阵 ----------
        t0 = time.time()
        hcp_dist_matrix = np.zeros((Ktrain, Ktrain))
        for i in range(Ktrain):
            xT = torch.from_numpy(train_3D[i]).float()
            for j in range(i + 1, Ktrain):
                yT = torch.from_numpy(train_3D[j]).float()
                loss = 0
                for g in gsw[2]:
                    loss += g.gsw(xT, yT)
                hcp_dist_matrix[i, j] = loss.item()

        hcp_dist_matrix = hcp_dist_matrix + hcp_dist_matrix.T
        t1 = time.time()
        dt_train_train = t1 - t0
        print("Time for train-train GSW distances:", dt_train_train, "seconds")

        # ---------- 2) 测试–训练 HCP 距离矩阵 ----------
        t2 = time.time()
        hcp_dist_matrix2 = np.zeros((Ktest, Ktrain))
        for i in range(Ktest):
            xT = torch.from_numpy(test_3D[i]).float()
            for j in range(Ktrain):
                yT = torch.from_numpy(train_3D[j]).float()
                loss = 0
                for g in gsw[2]:
                    loss += g.gsw(xT, yT)
                hcp_dist_matrix2[i, j] = loss.item()
        t3 = time.time()
        dt_test_train = t3 - t2
        print("Time for test-train GSW distances:", dt_test_train, "seconds")

        # ---------- 3) KNN 拟合 + 预测 ----------
        t4 = time.time()
        estimator = KNeighborsClassifier(metric='precomputed', n_neighbors=5)
        estimator.fit(hcp_dist_matrix, train_label)

        y_pred = estimator.predict(hcp_dist_matrix2)
        acc = np.sum(y_pred == test_label) / Ktest
        print('Accuracy is ', acc, '%')

        t5 = time.time()
        dt_knn = t5 - t4
        dt_total = t5 - t_total
        print("Time for KNN fit + predict:", dt_knn, "seconds")
        print("Total time:", dt_total, "seconds")

        # ---------- 4) 记录本轮结果 ----------
        acc_list.append(acc)
        time_train_train_list.append(dt_train_train)
        time_test_train_list.append(dt_test_train)
        time_knn_list.append(dt_knn)
        time_total_list.append(dt_total)

    # ================== 5) 汇总统计 ==================
    print("\n========== Summary over", n_runs, "runs ==========")
    print("Accuracy: min = {:.4f}, max = {:.4f}, mean = {:.4f}".format(
        min(acc_list), max(acc_list), sum(acc_list) / len(acc_list)
    ))

    print("Total time: min = {:.3f}s, max = {:.3f}s, mean = {:.3f}s".format(
        min(time_total_list), max(time_total_list), sum(time_total_list) / len(time_total_list)
    ))

    print("Train-train GSW time: min = {:.3f}s, max = {:.3f}s, mean = {:.3f}s".format(
        min(time_train_train_list), max(time_train_train_list), sum(time_train_train_list) / len(time_train_train_list)
    ))

    print("Test-train GSW time: min = {:.3f}s, max = {:.3f}s, mean = {:.3f}s".format(
        min(time_test_train_list), max(time_test_train_list), sum(time_test_train_list) / len(time_test_train_list)
    ))

    print("KNN fit+predict time: min = {:.3f}s, max = {:.3f}s, mean = {:.3f}s".format(
        min(time_knn_list), max(time_knn_list), sum(time_knn_list) / len(time_knn_list)
    ))




if __name__ == "__main__":
    main()