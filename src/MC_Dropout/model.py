from src.priors import *
from src.base_net import *

import torch
import torch.nn as nn
import torch.nn.functional as F


# %% 神经网络模型
# MC_Dropout网络
class MC_Dropout_Net(nn.Module):
    def __init__(self, layers, pdrop=0.5, init_out_noise=0.0):
        super(MC_Dropout_Net, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layers[i-1], layers[i]) for i in range(1, len(layers))])
        self.p = pdrop

        self.act = nn.Tanh()

        self.out_noise = nn.Parameter(torch.Tensor([init_out_noise]))  # 模型输出的标准差的softplus参数。为计算似然函数，模型输出需是一个分布，
                                                                        # forward输出的值相当于只是这个分布的均值，而其标准差由log_noise决定

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.dropout(x, p=self.p, training=True, inplace=True)
            x = self.act(x)
        x = self.layers[-1](x)
        return x
    

# MCDropout方法的贝叶斯神经网络模型
class MCDropout_BayesModel(BaseNet):
    def __init__(self, layers, pdrop, lamda, prior_class=isotropic_gauss_prior(mu=0, sigma=0.1), learning_rate=1e-3, device='cpu'):               # 网络结构，dropout概率，先验分布，学习率，gpu或cpu
        super(MCDropout_BayesModel, self).__init__()
        self.layers = layers
        self.pdrop = pdrop
        self.prior_std = prior_class.sigma
        self.lr = learning_rate
        self.device = device
        self.model_dir = './model'
        self.model_path = self.model_dir + '/mcdropout_bayesmodel.pth'           # 模型保存路径
        # self.lamda = (1-self.pdrop) / (2*self.prior_std**2)          # 正则项系数，根据KL散度计算公式得到，即使用了这个系数也和完整的公式差一个常数项，因此直接人为给定合适的lamda
        self.lamda = lamda

        self.net = MC_Dropout_Net(self.layers, self.pdrop)
        self.net.to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        self.loss_fn = isotropic_gauss_loglike    # 这里可以换成下方的loss_fn函数，舍弃掉常数项并将output_noise默认为1减少计算量

        mkdir(self.model_dir)   # 创建模型保存文件夹

        # 训练日志
        self.comp_log = []
        self.llhd_log = []
        self.loss_log = []
        self.err_log = []

        self.epoch = 0

        print("layers: ", self.layers)

    # def loss_fn(self, y, output):
    #     return (-0.5*(y-output)**2).sum()

    def compute_l2_regularization_exclude(self, exclude_names=('out_noise',)):
        l2_reg = 0.0
        for name, param in self.net.named_parameters():
            if param.requires_grad and not any(ex in name for ex in exclude_names):
                l2_reg += torch.sum(param ** 2)
        return l2_reg


    # 用VI算法训练模型
    def train(self, train_dataset, epochs, mc_samples=3):
        eps = 1e6
        Nbatch = len(train_dataset)       # 训练集的batch数，小批量训练时计算复杂性代价comp需要除以batch数

        for self.epoch in range(epochs):
            train_loader = iter(train_dataset)
            # 记录每个epoch的平均loss
            loss_epoch = 0
            comp_epoch = 0
            llhd_epoch = 0
            error_epoch = 0
            for x,y in train_loader:
                x, y = torch.from_numpy(x).type(torch.float32), torch.from_numpy(y).type(torch.float32)
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                comp = 0    # 复杂性代价，用于表示近似后验分布和先验分布的契合程度
                llhd = 0    # 似然代价，用于表示对训练数据的拟合程度
                # 用蒙特卡洛法估计期望，从而计算comp和llhd
                for i in range(mc_samples):
                    output = self.net(x)
                    output_noise = 1e-6 + F.softplus(self.net.out_noise, beta=1, threshold=20)
                    llhd_i = self.loss_fn(y, output, output_noise)
                    llhd += llhd_i
                comp = self.lamda * self.compute_l2_regularization_exclude() / Nbatch
                llhd /= mc_samples

                loss = comp - llhd
                loss.backward()
                self.optimizer.step()
                self.schedule.step()
                self.lr = self.optimizer.param_groups[0]['lr']

                # error_epoch += F.mse_loss(output, y).item()
                error_epoch += (torch.norm(output-y) / torch.norm(y)).item()
                loss_epoch += loss.item()
                comp_epoch += comp.item()
                llhd_epoch += llhd.item()

            loss_epoch = loss_epoch / Nbatch
            comp_epoch = comp_epoch / Nbatch
            llhd_epoch = llhd_epoch / Nbatch
            error_epoch = error_epoch / Nbatch

            if self.epoch % 500 == 0:
                print('Epoch: %d, Loss: %.4f, Comp: %.4f, LLHD: %.4f, Error: %.4f' % (self.epoch, loss_epoch, comp_epoch, llhd_epoch, error_epoch))

                if error_epoch < eps:
                    eps = error_epoch
                    self.save(self.model_path)
                    print('Best model saved! Best Error: %.4f\n' % eps)

            self.comp_log.append(comp_epoch)
            self.llhd_log.append(llhd_epoch)
            self.loss_log.append(loss_epoch)
            self.err_log.append(error_epoch)
    

    # 分布估计，输出预测值的均值和方差，以及预测误差
    def test_distribution_estimate(self, test_dataset, mc_samples=100):
        x, y = test_dataset.x, test_dataset.y
        x, y = torch.from_numpy(x).type(torch.float32), torch.from_numpy(y).type(torch.float32)
        x, y = x.to(self.device), y.to(self.device)
        output = []
        for i in range(mc_samples):
            output_i, _, _ = self.net(x)
            output.append(output_i)
        output = torch.stack(output, dim=0)
        output_mean = torch.mean(output, dim=0)
        output_var = torch.var(output, dim=0)
        output_noise = 1e-6 + F.softplus(self.net.out_noise, beta=1, threshold=20)
        error = F.mse_loss(output_mean, y).item()
        print('Error: %.4f' % error)

        return output_mean, output_var, output_noise, error
    

    # 输入x，输出预测值的均值和方差
    def predict(self, x, mc_samples=100):
        x = torch.from_numpy(x).type(torch.float32)
        x = x.to(self.device)
        output = []
        for i in range(mc_samples):
            output_i = self.net(x)
            # print("output_i shape: ", output_i.shape)
            output.append(output_i)
        output = torch.stack(output, dim=0)
        # print("output shape: ", output.shape)
        output_mean = torch.mean(output, dim=0)
        output_var = torch.var(output, dim=0)
        output_noise = 1e-6 + F.softplus(self.net.out_noise, beta=1, threshold=20)

        return output_mean, output_var, output_noise
    