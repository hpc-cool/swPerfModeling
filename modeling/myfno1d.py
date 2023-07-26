import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functools import partial
from timeit import default_timer
from utilities3 import *
from Adam import Adam
import matplotlib as mpl
import netron
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

torch.manual_seed(0)
np.random.seed(0)


def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    op = partial(torch.einsum, "bix,iox->box")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)


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
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

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
        self.padding = 128  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding]  # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


def plot(model, x_train, y_train, x_test, y_test, multiple, multiple_x):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    y_test *= multiple
    x_test *= multiple_x
    ax1.plot(x_test[:, 0, 0], y_test, color='b', linestyle='-.', marker='.', label='测试数据')

    l = min(x_train).numpy()[0][0] * multiple_x
    r = max(x_test).numpy()[0][0]

    index = 0
    x_test = np.linspace(l, r, 200)
    y_test = x_test
    x_test = x_test.reshape(200, 1, 1)
    x_test /= multiple_x
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                              batch_size=1,
                                              shuffle=False)

    pred = torch.zeros(y_test.shape)
    with torch.no_grad():
        for x, y in test_loader:
            # test_l2 = 0
            out = model(x).view(-1)
            pred[index] = out
            # test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            index = index + 1

    # scipy.io.savemat('pred/burger_test.mat', mdict={'pred': pred.cpu().numpy()})
    idx = np.argsort(x_test[:, 0, 0])
    X = x_test[idx, 0]

    X *= multiple_x
    pred *= multiple
    ax1.plot(X, pred[idx], "-r", marker='', label="实际拟合")

    x_train *= multiple_x
    y_train *= multiple
    ax1.plot(x_train[:, 0], y_train[:, 0], color='k', linestyle='-.', marker='.', label='训练数据')
    # plt.title(f'FNO拟合 %.3f s' % alltime)

    plt.xlabel("num of procs")
    plt.ylabel("runtime")
    plt.legend()
    # plt.show()

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


def get_mape(records_real, records_predict):
    if len(records_real) == len(records_predict):
        return sum([np.fabs(x - y) / x for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


def run_train(epochs, model, optimizer, scheduler, ntrain, ntest, s):
    # dataset
    in_x = np.double([4, 6, 16, 24, 36, 48, 52, 64, 84, 96])
    in_y = np.double([8224, 8853, 8438, 8401, 8863, 8712, 8279, 8218, 8676, 9287])
    in_y = np.double(
        [44735.66, 35398.79, 18342.92, 16025.91, 14181.99, 13178.13, 12673.41, 11941.68, 11506.08, 11520.12])

    #
    # in_x = np.double([300, 400, 512, 600, 768, 860, 1024, 1280, 1560, 2048])
    # in_y = np.double([62615.1624, 50552.02311, 44416.22043, 40371.70171, 34995.73124,
    #                   33475.63041, 32818.65864, 30781.08843, 29739.81095, 29866.03951])

    in_x = np.double([600, 684, 720, 800, 860, 900, 1024, 1280, 1360, 1420])
    in_y = np.double(
        [19561787520, 17441061339, 18103654423, 16778475999, 16123322323, 15291971036, 14248585115, 13898926964,
         15166985932, 13968659680])
    in_y = np.double([66561, 60767, 59402, 55814, 55019, 50476, 48663, 45285, 45252, 44016])
    # in_x = np.double([16, 30, 64, 128, 192, 256, 320, 384, 448, 512, 768, 1024])
    # in_y = np.double(
    #     [786319070, 3170618940, 64711492690, 96932501590, 96925054810, 94732711140, 99341121830, 95961049500,
    #      98806139310, 97195380000, 1.02753E+11, 1.15015E+11])
    #
    # in_x = np.double([30, 64, 128, 192, 256, 384, 448, 512, 1024, 320, 768])
    # in_y = np.double(
    #     [75681879530, 44162016220, 43911349330, 15538938030, 23051554850, 15034477190, 12430370000, 5741724690,
    #      5426658540, 17742253570, 3616942820])
    #
    # # NPB_SP
    # in_x = np.double([36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289])
    # in_y = np.double(
    #     [31.71, 20.89, 14.39, 12.2, 9.88, 8.89, 7.65, 6.31, 5.93, 5.28, 5.08, 4.64])
    # # NPB_CG
    # in_x = np.double([16,
    #                   32,
    #                   64,
    #                   128,
    #                   256])
    # in_y = np.double(
    #     [1340.448196,
    #      869.8532233,
    #      901.4933853,
    #      311.1356282,
    #      145.1259021])
    # #NPB_BT
    # in_x = np.double([25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225])
    #
    # in_y = np.double(
    #     [1906.538612, 1820.201476, 1585.139254, 1223.467914, 1066.485075, 872.8975077, 733.8582774,
    #      696.3585586, 534.3330822, 499.526353, 460.8528527])

    in_x = np.double([32, 64, 96, 120,
                      144, 160, 192, 240,
                      288, 320, 480, 720,
                      960])
    in_y = np.double([167, 121, 101, 94,
                      83, 83, 83, 78,
                      73, 78, 95, 115,
                      163])
    # in_y = np.double([427.23, 259.28, 144.94, 107.02,
    #                   90.976, 88.718, 87.469, 57.637,
    #                   51.395, 52.766, 32.938, 23.175,
    #                   18.09])
    # in_y = np.double([104.33, 56.06, 40.07, 32.977,
    #                   28.341, 26.437, 22.548, 19.004,
    #                   16.332, 15.05, 10.659, 8.0433,
    #                   6.4496])
    # normalize
    mean = np.mean(in_y)
    multiple = 1
    while mean > 10:
        mean /= 10
        multiple *= 10
    in_y /= multiple

    mean_x = np.mean(in_x)
    multiple_x = 1
    # while mean_x > 10:
    #     mean_x /= 10
    #     multiple_x *= 10
    in_x /= multiple_x
    # sort
    x_index = np.argsort(in_x)
    in_x = in_x[x_index]
    in_y = in_y[x_index]
    # 划分训练集和测试集
    true_x = in_x[:ntrain * s]
    true_y = in_y[:ntrain]
    x_test = in_x[ntrain * s:]
    y_test = in_y[ntrain:]
    # [7,]->[7,1]
    x_train = true_x[:, np.newaxis]
    y_train = true_y[:, np.newaxis]
    # reshape
    x_train = x_train.reshape(ntrain, s, 1)
    x_test = x_test.reshape(ntest, s, 1)
    # numpy->tensor
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()
    # data loader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=ntrain,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=ntest,
        shuffle=False)
    # define loss
    myloss = LpLoss(size_average=False)
    alltime = 0
    train_losses = []
    test_losses = []
    train_mses = []
    test_mses = []

    print('train start')

    for ep in range(epochs):
        t1 = default_timer()
        model.train()
        train_mse = 0
        train_l2 = 0
        train_mape = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)

            mse = F.mse_loss(out.view(ntrain, -1), y.view(ntrain, -1), reduction='mean')
            l2 = myloss(out.view(ntrain, -1), y.view(ntrain, -1))
            l2.backward()  # use the l2 relative loss
            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()
            train_mape += get_mape(y.view(ntrain, -1).detach().numpy(),
                                   out.view(ntrain, -1).detach().numpy())

        t2 = default_timer()
        scheduler.step()

        model.eval()
        test_l2 = 0.0
        test_mse = 0.0
        test_mape = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x)
                test_l2 += myloss(out.view(ntest, -1),
                                  y.view(ntest, -1)).item()
                test_mse += F.mse_loss(out.view(ntest, -1),
                                       y.view(ntest, -1),
                                       reduction='mean').item()
                test_mape += get_mape(y.view(ntest, -1).detach().numpy(),
                                      out.view(ntest, -1).detach().numpy())

        train_l2 /= ntrain
        train_mse /= len(train_loader)
        train_mape /= len(train_loader)
        test_l2 /= ntest
        test_mse /= len(test_loader)
        test_mape /= len(test_loader)
        train_mses.append(train_mape)
        train_losses.append(train_l2)
        test_mses.append(test_mape)
        test_losses.append(test_l2)

        alltime += t2 - t1
        # print(ep, t2 - t1, train_mse, train_l2, test_l2)
        if train_l2 - 1e-2 < 0 or ep == epochs - 1:
            print('train_mse:', train_mse, 'train_l2:', train_l2, 'train_mape:', train_mape, 'epochs:', ep)
            print('test_mse:', test_mse, 'test_l2:', test_l2, 'test_mape', test_mape)
            break

    with torch.no_grad():
        allsum = 0
        alllen = 0
        for x, y in train_loader:
            out = model(x).view(-1).numpy()
            y = y.view(-1).numpy()
            allsum += sum(np.fabs(y - out) / y)
            alllen += len(y)
            print(out * multiple)

            print("TrainSet Relative Error:", (np.fabs(y - out) / y))

        for x, y in test_loader:
            out = model(x).view(-1).numpy()
            y = y.view(-1).numpy()
            allsum += sum(np.fabs(y - out) / y)
            alllen += len(y)
            print(out * multiple)
            print("TestSet Relative Error:", (np.fabs(y - out) / y))
        print("Mean Relative Error:", allsum / alllen)

    print('Training time... ', alltime)
    plot(model, x_train, y_train, x_test, y_test, multiple, multiple_x)

    # onnx_path="onnx_model_name.onnx"
    # torch.onnx.export(model, x_train, onnx_path)  # 导出神经网络模型为onnx格式
    # netron.start(onnx_path) # 启动netro


def main():
    ntrain = 5
    ntest = 8
    s = 1

    batch_size = 7
    learning_rate = 0.01
    epochs = 300
    step_size = 10
    gamma = 0.8
    modes = 1
    width = 16

    model = FNO1d(modes, width)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    print("model params:", count_params(model))
    run_train(epochs, model, optimizer, scheduler, ntrain, ntest, s)


if __name__ == '__main__':
    main()
