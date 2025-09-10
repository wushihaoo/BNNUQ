from src.priors import *
from src.base_net import *
from .hmc_optimizer import *

import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
# 用HMC优化的贝叶斯神经网络
class HMC_BayesNet(nn.Module):
    def __init__(self, layers, init_out_noise=0.0):
        super(HMC_BayesNet, self).__init__()
        
        self.layers = nn.ModuleList([nn.Linear(layers[i-1], layers[i]) for i in range(1, len(layers))])

        self.act = nn.Tanh()

        self.out_noise = nn.Parameter(torch.Tensor([init_out_noise]))  # 模型输出的标准差的softplus参数。为计算似然函数，模型输出需是一个分布，
                                                                        # forward输出的值相当于只是这个分布的均值，而其标准差由log_noise决定

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act(x)
        x = self.layers[-1](x)

        return x
    

# 用HMC优化的贝叶斯神经网络模型
class HMC_BayesModel(BaseNet):
    def __init__(self, layers, prior_class=isotropic_gauss_prior(mu=0, sigma=0.1), lr=0.01, n_steps=10, device='cpu'):               # 网络结构，先验分布，学习率，gpu或cpu
        super(HMC_BayesModel, self).__init__()
        self.layers = layers
        self.prior = prior_class
        self.lr = lr
        self.n_steps = n_steps
        self.device = device
        self.model_dir = './model/hmc_net_samples'
        self.model_path = self.model_dir + '/hmc_bayesmodel_0.pth'           # 模型保存路径
        self.net_id = 0
        self.epoch = 0
        
        self.net = HMC_BayesNet(self.layers)
        self.net.to(self.device)

        self.optimizer = HMC_Optimizer_nlp(self.net.parameters(), self.lr, self.n_steps)
        self.llhd = isotropic_gauss_loglike    # 这里可以换成下方的loss_fn函数，舍弃掉常数项减少计算量

        mkdir(self.model_dir)   # 创建模型保存文件夹

        # 训练日志
        self.loss_log = []
        self.err_log = []
        self.prior_log = []
        self.llhd_log = []

        print("layers: ", self.layers)

    # def llhd(self, y, output):
    #     return (-0.5*(y-output)**2).sum()

    # def prior(self, exclude_names=('out_noise',)):
    #     l2_reg = 0.0
    #     for name, param in self.net.named_parameters():
    #         if param.requires_grad and not any(ex in name for ex in exclude_names):
    #             l2_reg += torch.sum(param ** 2)
    #     return l2_reg
    
    def total_loss(self, y, output):
        output_noise = 1e-6 + F.softplus(self.net.out_noise, beta=1, threshold=20)
        log_prior = sum(self.prior.loglike(p) for p in self.net.parameters() if p.requires_grad)
        log_llhd = self.llhd(y, output, output_noise)
        return -log_prior -log_llhd


    # 用HMC算法采样模型参数
    def train(self, train_dataset, burn_in_epochs, sample_nets, mix_epochs):
        Nbatch = len(train_dataset)       # 训练集的batch数，小批量训练时计算复杂性代价comp需要除以batch数
        t_epochs = burn_in_epochs + sample_nets*mix_epochs

        for self.epoch in range(t_epochs):
            train_loader = iter(train_dataset)
            # 记录每个epoch的平均loss
            loss_epoch = 0
            error_epoch = 0
            prior_epoch = 0
            llhd_epoch = 0
            for x,y in train_loader:
                x, y = torch.from_numpy(x).type(torch.float32), torch.from_numpy(y).type(torch.float32)
                x, y = x.to(self.device), y.to(self.device)

                closure = make_closure(self.net, x, y, self.total_loss)
                self.optimizer.zero_grad()

                output = self.net(x)

                output_noise = 1e-6 + F.softplus(self.net.out_noise, beta=1, threshold=20)
                log_prior = sum(self.prior.loglike(p) for p in self.net.parameters() if p.requires_grad)
                log_llhd = self.llhd(y, output, output_noise)

                loss = self.total_loss(y, output)
                # loss = -log_llhd
                loss.backward()
                self.optimizer.step(closure)

                # error_epoch += F.mse_loss(output, y).item()
                error_epoch += (torch.norm(output-y) / torch.norm(y)).item()
                loss_epoch += loss.item()
                prior_epoch += log_prior.item()
                llhd_epoch += log_llhd.item()

            loss_epoch = loss_epoch / Nbatch
            error_epoch = error_epoch / Nbatch
            prior_epoch = prior_epoch / Nbatch
            llhd_epoch = llhd_epoch / Nbatch

            if self.epoch % 100 == 0:
                if self.epoch < burn_in_epochs:
                    print('Burn_in Epoch: %d, Posterior: %.4f, NLL: %.4f, Error: %.4f' % (self.epoch, loss_epoch, -llhd_epoch, error_epoch))
                else:
                    print('Sample Epoch: %d, Posterior: %.4f, NLL: %.4f, Error: %.4f' % (self.epoch, loss_epoch, -llhd_epoch, error_epoch))
                rates = self.optimizer.get_acceptance_rates()
                print('Acceptance Rate: ', [round(value, 4) for value in rates.values()])

            if (self.epoch >= burn_in_epochs) and (self.epoch % mix_epochs == 0):
                self.model_path = self.model_dir + '/hmc_bayesmodel_%d' % self.net_id + '.pth'
                self.save(self.model_path)
                self.net_id += 1

            self.loss_log.append(loss_epoch)
            self.err_log.append(error_epoch)
            self.prior_log.append(prior_epoch)
            self.llhd_log.append(llhd_epoch)
    
    
    # 输入x，输出预测值的均值和方差
    def predict(self, x):
        x = torch.from_numpy(x).type(torch.float32)
        x = x.to(self.device)
        output = self.net(x)
        output_noise = 1e-6 + F.softplus(self.net.out_noise, beta=1, threshold=20)

        return output, output_noise
    