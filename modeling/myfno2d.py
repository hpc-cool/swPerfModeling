'''use pytorch1.5'''
import os

import numpy as np
from matplotlib import cm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from sklearn import preprocessing

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)


# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False


def gemm(a, b):
    sizea = a.size()
    sizeb = b.size()
    out = torch.zeros(sizea[0], sizeb[1], sizea[2])
    for i in range(sizea[0]):
        for k in range(sizeb[1]):
            for p in range(sizea[2]):
                for j in range(sizea[1]):
                    out[i][k][p] += a[i][j][p] * b[j][k][p]
    return out


# Complex multiplication
def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    op = partial(torch.einsum, "bix,iox->box")
    x1 = op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1])
    x2 = op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    return torch.stack([x1, x2], dim=-1)
    # return torch.stack([
    #     op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
    #     op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    # ], dim=-1)


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 1, normalized=True, onesided=True)
        # print("x_ft", x_ft.shape)

        # Multiply relevant Fou1rier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1] = compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.irfft(out_ft, 1, normalized=True, onesided=True, signal_sizes=(x.size(-1),))
        return x


class SimpleBlock1d(nn.Module):
    def __init__(self, modes, width):
        super(SimpleBlock1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)
        self.padding = 63  # pad the domain if input is non-periodic

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        # self.conv0 = nn.Conv1d(self.width, self.width, 1)
        # self.conv1 = nn.Conv1d(self.width, self.width, 1)
        # self.conv2 = nn.Conv1d(self.width, self.width, 1)
        # self.conv3 = nn.Conv1d(self.width, self.width, 1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)  # 交换1、2维
        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]  # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.squeeze(x)
        x = self.fc3(x)
        return x


class Net1d(nn.Module):
    def __init__(self, modes, width):
        super(Net1d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock1d(modes, width)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.conv1(x)
        return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


def plot(model, x_train, y_train, x_test, y_test):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    # y_test = (y_test - 0.01) * (Max - Min) + Min
    # y_test = y_test * std + mean
    # y_test *= multiple
    ax1.plot(x_test[:, 0, 0], y_test, color='b', linestyle='-.', marker='.', label='测试数据')

    index = 0
    x_test = np.linspace(1, 2048, 200)
    y_test = x_test
    x_test = x_test.reshape(200, 1, 1)
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                              batch_size=1,
                                              shuffle=False)

    pred = torch.zeros(y_test.shape)
    with torch.no_grad():
        for x, y in test_loader:
            test_l2 = 0
            out = model(x).view(-1)
            pred[index] = out

            # test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            # print(index, test_l2)
            index = index + 1

    print(x_test, pred)
    # scipy.io.savemat('pred/burger_test.mat', mdict={'pred': pred.cpu().numpy()})
    idx = np.argsort(x_test[:, 0, 0])
    X = x_test[idx, 0]

    # pred *= multiple
    ax1.plot(X, pred[idx], "-r", marker='', label="实际拟合")
    # y_train *= multiple
    ax1.plot(x_train[:, 0], y_train[:, 0], color='k', linestyle='-.', marker='.', label='训练数据')
    # plt.title(f'FNO拟合 %.3f s' % alltime)
    plt.xlabel("num of procs")
    plt.ylabel("runtime")
    plt.legend()
    plt.show()

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(2, 1, 1)
    # ax2.plot(np.arange(len(train_losses)), train_losses, c='r', label='train_losses')
    # ax2.plot(np.arange(len(test_losses)), test_losses, c='b', label='test_losses')
    # plt.legend()
    # ax2 = fig2.add_subplot(2, 1, 2)
    # ax2.plot(np.arange(len(train_mses)), train_mses, c='r', linestyle='-.', label='train_mses')
    # ax2.plot(np.arange(len(test_mses)), test_mses, c='b', linestyle='-.', label='test_mses')
    # plt.yscale('log')
    # plt.xlabel('epoch')
    # plt.legend()
    # plt.show()


def plot2d():
    vegetables = [16, 32, 48, 64]
    farmers = [400, 600, 800, 1000, 1200]

    harvest = np.array([[88286523, 61568070, 44882850, 37082294, 30972732],
                        [44284398, 30472717, 21652564, 18617086, 15581054],
                        [34395475, 24075828, 17895584, 16015620, 13457141],
                        [28804107, 20288524, 15459552, 14007991, 12404582]])
    harvest = harvest / 10000000

    fig, ax = plt.subplots()
    im = ax.imshow(harvest)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(farmers)), labels=farmers)
    ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.show()


def plot3(pred, train_y, ss):
    x = np.double([16, 32, 48, 64])
    y = np.double([400, 600, 800, 1000, 1200])
    # y = np.double([20, 40, 60, 80, 100])

    x = np.double([64, 102, 162])
    y = np.double([36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289])

    X, Y = np.meshgrid(x, y)
    pred = pred[:, np.newaxis]
    # train_y = ss.inverse_transform(train_y)
    # pred = ss.inverse_transform(pred)

    vmin = min(np.min(pred), np.min(train_y))
    vmax = max(np.max(pred), np.max(train_y))
    norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
    fig = plt.figure(figsize=(4, 8))

    bx = fig.add_subplot(211)
    ax = fig.add_subplot(212)

    # 理论值画图
    c = bx.contourf(X, Y, train_y.reshape(12, 3), 60, norm=norm, cmap="rocket_r")
    # bx.set_xlabel('number of SPEs in CG')
    bx.set_ylabel('number of procs')
    bx.set_title('Actual')

    # 预测值画图
    f = ax.contourf(X, Y, pred.reshape(12, 3), 60, norm=norm, cmap="rocket_r")
    # ax.set_xlabel('number of SPEs in CG')
    ax.set_ylabel('number of procs')
    ax.set_title('Predicted')

    plt.xlabel('number of threads')
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    fig.colorbar(c, ax=[ax, bx], orientation='horizontal', pad=0.08, label='Times(s)', format='%.0f')

    # 采样点画图
    # plt.scatter(X, Y, s=40, c='none', edgecolors='k')
    # plt.show()

    # 理论值与预测值的均方误差
    print(np.average(np.fabs((pred - train_y) / train_y)))
    # print(np.mean(np.square(pred - train_y)))


def run_train(epochs, model, optimizer, scheduler, ntrain, ntest, s):
    in_x = np.double([300, 64, 400, 64, 512, 64, 600, 64, 768, 64,
                      860, 64, 1024, 64, 1280, 64, 1560, 64, 2048, 64])
    in_y = np.double([62615.1624, 50552.02311, 44416.22043, 40371.70171, 34995.73124,
                      33475.63041, 32818.65864, 30781.08843, 29739.81095, 29866.03951])

    in_x = np.double([30, 64, 128, 192, 256, 384, 448, 512, 1024, 320, 768])
    in_y = np.double(
        [75681879530, 44162016220, 43911349330, 15538938030, 23051554850, 15034477190, 12430370000, 5741724690,
         5426658540, 17742253570, 3616942820])

    in_x = np.double(
        [16, 400, 16, 600, 16, 800, 16, 1000, 16, 1200, 32, 400, 32, 600, 32, 800, 32, 1000, 32, 1200, 48, 400, 48, 600,
         48, 800, 48, 1000, 48, 1200, 64, 400, 64, 600, 64, 800, 64, 1000, 64, 1200])
    # kernel_host_7004053664364030334 plot2
    in_y = np.double(
        [88286523, 61568070, 44882850, 37082294, 30972732, 44284398, 30472717, 21652564, 18617086, 15581054, 34395475,
         24075828, 17895584, 16015620, 13457141, 28804107, 20288524, 15459552, 14007991, 12404582])

    # MPI_Recv
    in_y = np.double(
        [1167082133, 832809281, 678426610, 661868354, 495629271, 1012177904, 1062923998, 770891509, 562213991,
         510266110, 1017619232, 844790945, 703968037, 599589112, 477177429, 1159593242, 1031235839, 1457973578,
         709345833, 517635888])

    # MPI_Gather
    in_y = np.double(
        [1814790, 1797249, 1783503, 1710733, 1674093, 1814359, 1848587, 1786637, 1701746, 1670291, 1842008, 1836166,
         1841166, 1711574, 1695301, 1838440, 1880766, 1820680, 1746587, 1693003])

    # 较大规模 plot1
    in_y = np.double([
        65.372, 49.835, 41.01, 36.641, 35.235,
        56.727, 44.117, 36.145, 32.438, 32.044,
        52.348, 41.268, 34.857, 31.464, 31.538,
        51.921, 42.332, 34.824, 31.254, 31.476,
    ])

    # 较小规模
    # in_x = np.double(
    #     [16, 20, 16, 40, 16, 60, 16, 80, 16, 100,
    #      32, 20, 32, 40, 32, 60, 32, 80, 32, 100,
    #      48, 20, 48, 40, 48, 60, 48, 80, 48, 100,
    #      64, 20, 64, 40, 64, 60, 64, 80, 64, 100])
    # in_y = np.double(
    #     [20.2544, 14.5504, 13.2972, 12.2572, 12.3722, 17.0316,
    #      13.2566, 12.7594, 12.0012, 11.6176, 17.0316,
    #      13.2566, 12.7594, 12.0012, 11.6176, 16.2124,
    #      13.1356, 12.1706, 11.8676, 11.555])



    in_x = np.double([
        64, 36, 102, 36, 162, 36,
        64, 49, 102, 64, 162, 49,
        64, 64, 102, 64, 162, 64,
        64, 81, 102, 81, 162, 81,
        64, 100, 102, 100, 162, 100,
        64, 121, 102, 121, 162, 121,
        64, 144, 102, 144, 162, 144,
        64, 169, 102, 169, 162, 169,
        64, 196, 102, 196, 162, 196,
        64, 225, 102, 225, 162, 225,
        64, 256, 102, 256, 162, 256,
        64, 289, 102, 289, 162, 289,

    ])

    in_y = np.double([
        6.31, 31.71, 137.86,
        4.95, 20.89, 93.66,
        4.11,
        14.39, 77.33,
        3.38,
        12.2, 59.14,
        2.99,
        9.88, 45.69,
        2.72,
        8.89, 34.52,
        2.78,
        7.65, 34.05,
        2.04,
        6.31, 22.88,
        2.13,
        5.93, 19.25,
        2.01,
        5.28, 17.55,
        1.93,
        5.08, 15.70,
        1.84,
        4.64, 14.82
    ])

    in_x = in_x.reshape(ntrain + ntest, s)
    in_y = in_y.reshape(ntrain + ntest, 1)
    ss = preprocessing.RobustScaler()
    # in_x = ss.fit_transform(in_x)
    # in_y = ss.fit_transform(in_y)

    # normalize
    mean = np.mean(in_y)
    multiple = 1
    while mean > 10:
        mean /= 10
        multiple *= 10
    in_y /= multiple

    mean_x = np.mean(in_x)
    multiple_x = 1
    while mean_x > 10:
        mean_x /= 10
        multiple_x *= 10
    in_x /= multiple_x

    # 划分训练集和测试集
    true_x = in_x[:ntrain, :]
    true_y = in_y[:ntrain, :]
    x_test = in_x[ntrain:, :]
    y_test = in_y[ntrain:, :]
    # [7,]->[7,1]
    x_train = true_x[:, np.newaxis]
    y_train = true_y[:, np.newaxis]
    # x_train = in_x[:ntrain * s, np.newaxis]
    # y_train = in_y[:ntrain, np.newaxis]
    # x_test = in_x[ntrain * s:, np.newaxis]
    # y_test = in_y[ntrain:, np.newaxis]

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    x_train = x_train.reshape(ntrain, s, 1)
    x_test = x_test.reshape(ntest, s, 1)

    # cat the locations information
    # grid = np.linspace(0, 2 * np.pi, 1).reshape(1, 1, 1)
    # grid = torch.tensor(grid, dtype=torch.float)
    # x_train = torch.cat([x_train.reshape(ntrain, 1, 1), grid.repeat(ntrain, 1, 1)], dim=2)
    # x_test = torch.cat([x_test.reshape(ntest, 1, 1), grid.repeat(ntest, 1, 1)], dim=2)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=ntrain,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=ntest,
                                              shuffle=False)

    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            # print(out.view(ntrain,-1))
            mse = F.mse_loss(out.view(ntrain, -1), y.view(ntrain, -1), reduction='mean')
            # mse = F.mse_loss(out, y, reduction='mean')
            # mse.backward()
            l2 = myloss(out.view(ntrain, -1), y.view(ntrain, -1))
            l2.backward()  # use the l2 relative loss

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

            scheduler.step()
            model.eval()
            test_l2 = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    out = model(x)
                    test_l2 += myloss(out.view(ntest, -1), y.view(ntest, -1)).item()

            train_mse /= len(train_loader)
            train_l2 /= ntrain
            test_l2 /= ntest

            t2 = default_timer()
            # print(ep, t2 - t1, train_mse, train_l2, test_l2)

    with torch.no_grad():
        for x, y in train_loader:
            train_out = model(x).view(-1).numpy()
            train_y = y.view(-1).numpy()
            # print(out * multiple)
            print(train_out * multiple)
            print(train_y * multiple)
            train_mae = np.fabs((train_y - train_out) / train_y)
            print("TrainSet Relative Error:", np.average(train_mae), max(train_mae), min(train_mae))

        for x, y in test_loader:
            test_out = model(x).view(-1).numpy()
            test_y = y.view(-1).numpy()
            print(test_out * multiple)
            print(test_y * multiple)
            test_mae = np.fabs((test_y - test_out) / test_y)
            print("TestSet Relative Error:", np.average(test_mae), max(test_mae), min(test_mae))

        print(",".join(str(i) for i in np.concatenate([train_mae, test_mae])))
        train_out = model(x_train).view(-1).numpy()
        test_out = model(x_test).view(-1).numpy()
    # plot(model, x_train, y_train, x_test, y_test)
    # plot2d()
    # plot3(np.concatenate([train_out, test_out]), np.concatenate([train_y, test_y]))
    # plot3(np.concatenate([train_out, test_out]) * multiple, in_y * multiple, ss)


def main():
    ntrain = 27
    ntest = 9

    s = 2
    learning_rate = 0.01

    epochs = 300
    step_size = 100
    gamma = 0.9
    modes = 16
    width = 32

    model = Net1d(modes, width)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    print(model.count_params())
    run_train(epochs, model, optimizer, scheduler, ntrain, ntest, s)


if __name__ == "__main__":
    main()
