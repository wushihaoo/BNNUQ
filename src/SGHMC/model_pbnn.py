from src.priors import *
from src.base_net import *
from .sghmc_optimizer import *

import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
# 用SGHMC优化的贝叶斯神经网络
class SGHMC_PBayesNet(nn.Module):
    def __init__(self, dnnlayers, bnnlayers, init_out_noise=0.0):
        super(SGHMC_PBayesNet, self).__init__()
        
        self.bnnlayers = nn.ModuleList([nn.Linear(bnnlayers[i-1], bnnlayers[i]) for i in range(1, len(bnnlayers))])
        self.dnnlayers = nn.ModuleList([nn.Linear(dnnlayers[i-1], dnnlayers[i]) for i in range(1, len(dnnlayers))])

        self.act = nn.Tanh()

        self.out_noise = nn.Parameter(torch.Tensor([init_out_noise]))  # 模型输出的标准差的softplus参数。为计算似然函数，模型输出需是一个分布，
                                                                        # forward输出的值相当于只是这个分布的均值，而其标准差由log_noise决定

    def forward(self, x):
        y_d = x
        for layer in self.dnnlayers[:-1]:
            y_d = layer(y_d)
            y_d = self.act(y_d)
        y_d = self.dnnlayers[-1](y_d)

        y_b = x
        for layer in self.bnnlayers[:-1]:
            y_b = layer(y_b)
            y_b = self.act(y_b)
        y_b = self.bnnlayers[-1](y_b)
        y_b = F.softmax(y_b, dim=-1)

        y = torch.sum(y_d * y_b, dim=-1).unsqueeze(-1)

        return y
    

# 用SGHMC优化的贝叶斯神经网络模型
class SGHMC_PBayesModel(BaseNet):
    def __init__(self, dnnlayers, bnnlayers, prior_class=isotropic_gauss_prior(mu=0, sigma=0.1), dnnlr=0.01, bnnlr=0.01, base_C=0.05, device='cpu'):               # 网络结构，先验分布，学习率，gpu或cpu
        super(SGHMC_PBayesModel, self).__init__()
        self.dnnlayers = dnnlayers
        self.bnnlayers = bnnlayers
        self.prior = prior_class
        self.dnnlr = dnnlr
        self.lr = bnnlr
        self.base_C = base_C
        self.device = device
        self.model_dir = './model/pbnnsghmc_net_samples'
        self.model_path = self.model_dir + '/pbnnsghmc_bayesmodel_0.pth'           # 模型保存路径
        self.net_id = 0
        self.epoch = 0
        
        self.net = SGHMC_PBayesNet(self.dnnlayers, self.bnnlayers)
        self.net.to(self.device)

        self.mapoptimizer = torch.optim.Adam(self.net.parameters(), lr=self.dnnlr)
        self.optimizer = SGHMC_Optimizer(list(self.net.bnnlayers.parameters())+[self.net.out_noise], self.lr, self.base_C)

        self.llhd = isotropic_gauss_loglike    # 这里可以换成下方的loss_fn函数，舍弃掉常数项减少计算量

        mkdir(self.model_dir)   # 创建模型保存文件夹

        # 训练日志
        self.map_loss_log = []
        self.loss_log = []
        self.err_log = []
        self.prior_log = []
        self.llhd_log = []

        print("dnn layers: ", self.dnnlayers)
        print("bnn layers: ", self.bnnlayers)

    # def llhd(self, y, output):
    #     return (-0.5*(y-output)**2).sum()

    # def prior(self, exclude_names=('out_noise',)):
    #     l2_reg = 0.0
    #     for name, param in self.net.named_parameters():
    #         if param.requires_grad and not any(ex in name for ex in exclude_names):
    #             l2_reg += torch.sum(param ** 2)
    #     return l2_reg
    
    def map_loss(self, y, output):
        # output_noise = 1e-6 + F.softplus(self.net.out_noise, beta=1, threshold=20)
        # log_prior = sum(self.prior.loglike(p) for p in self.net.parameters() if p.requires_grad)
        # log_llhd = self.llhd(y, output, output_noise)
        # return -log_prior -log_llhd
        # 带l2正则项的均方误差
        map_loss = F.mse_loss(output, y) + 1e-6*sum(torch.sum(p**2) for p in self.net.parameters() if p.requires_grad)
        # map_loss = F.mse_loss(output, y)
        return map_loss
    
    # 并不是准确的后验估计，先验计算舍掉了常数项，似然舍掉了Nbatch的乘数
    def bnn_loss(self, y, output):
        output_noise = 1e-6 + F.softplus(self.net.out_noise, beta=1, threshold=20)
        log_prior = sum(self.prior.loglike(p) for p in list(self.net.bnnlayers.parameters())+[self.net.out_noise] if p.requires_grad)
        log_llhd = self.llhd(y, output, output_noise)
        return -log_prior -log_llhd


    # 用SGHMC算法采样模型参数
    def train(self, train_dataset, map_epochs, burn_in_epochs, sample_nets, mix_epochs):
        Nbatch = len(train_dataset)       # 训练集的batch数，小批量训练时计算复杂性代价comp需要除以batch数
        t_epochs = map_epochs + burn_in_epochs + sample_nets*mix_epochs

        for self.epoch in range(t_epochs):
            train_loader = iter(train_dataset)
            # 记录每个epoch的平均loss
            map_loss_epoch = 0
            loss_epoch = 0
            error_epoch = 0
            prior_epoch = 0
            llhd_epoch = 0

            if self.epoch <= map_epochs:
                train_mode = 'map'
            else:
                train_mode = 'bnn'
                for p in self.net.dnnlayers.parameters():
                        p.requires_grad_(False)

            burn_in = (self.epoch < burn_in_epochs+map_epochs)
            resample_v = (self.epoch % 2 == 0)
            resample_prior = (self.epoch % 2 == 0)
            for x,y in train_loader:
                x, y = torch.from_numpy(x).type(torch.float32), torch.from_numpy(y).type(torch.float32)
                x, y = x.to(self.device), y.to(self.device)

                if train_mode == 'map':
                    self.mapoptimizer.zero_grad()
                    output = self.net(x)
                    loss = self.map_loss(y, output)
                    loss.backward()
                    self.mapoptimizer.step()
                    map_loss_epoch += loss.item()
                else:
                    self.optimizer.zero_grad()
                    output = self.net(x)
                    output_noise = 1e-6 + F.softplus(self.net.out_noise, beta=1, threshold=20)
                    log_prior = sum(self.prior.loglike(p) for p in list(self.net.bnnlayers.parameters())+[self.net.out_noise] if p.requires_grad)
                    log_llhd = self.llhd(y, output, output_noise)
                    # loss = -log_prior - log_llhd*Nbatch
                    loss = -log_llhd*Nbatch

                    loss.backward()
                    self.optimizer.step(burn_in=burn_in, resample_v=resample_v, resample_prior=resample_prior)

                    loss_epoch += loss.item()
                    prior_epoch += log_prior.item()
                    llhd_epoch += log_llhd.item()

                # error_epoch += F.mse_loss(output, y).item()
                error_epoch += (torch.norm(output-y) / torch.norm(y)).item()
                    
            map_loss_epoch = map_loss_epoch / Nbatch
            loss_epoch = loss_epoch / Nbatch
            error_epoch = error_epoch / Nbatch
            prior_epoch = prior_epoch / Nbatch
            llhd_epoch = llhd_epoch / Nbatch

            if (self.epoch <= map_epochs) and (self.epoch % 200 == 0):
                print('Map Epoch: %d, Map Loss: %.4f, Error: %.4f' % (self.epoch, map_loss_epoch, error_epoch))

            if (self.epoch > map_epochs) and (self.epoch % 200 == 0):
                if self.epoch < burn_in_epochs+map_epochs:
                    print('Burn_in Epoch: %d, NLL: %.4f, Error: %.4f' % (self.epoch, loss_epoch, error_epoch))
                else:
                    print('Sample Epoch: %d, NLL: %.4f, Error: %.4f' % (self.epoch, loss_epoch, error_epoch))
            
            if (self.epoch >= burn_in_epochs+map_epochs) and (self.epoch % mix_epochs == 0):
                self.model_path = self.model_dir + '/pbnnsghmc_bayesmodel_%d' % self.net_id + '.pth'
                self.save(self.model_path)
                self.net_id += 1

            if train_mode == 'map':
                self.map_loss_log.append(map_loss_epoch)
            else:
                self.loss_log.append(loss_epoch)
                self.prior_log.append(prior_epoch)
                self.llhd_log.append(llhd_epoch)

            self.err_log.append(error_epoch)
    
    
    # 输入x，输出预测值的均值和方差
    def predict(self, x):
        x = torch.from_numpy(x).type(torch.float32)
        x = x.to(self.device)
        output = self.net(x)
        output_noise = 1e-6 + F.softplus(self.net.out_noise, beta=1, threshold=20)

        return output, output_noise
    