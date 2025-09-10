from src.priors import *
from src.base_net import *

import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
# 用rMAP优化的贝叶斯神经网络
class rMAP_BayesNet(nn.Module):
    def __init__(self, layers, init_out_noise=0.0):
        super(rMAP_BayesNet, self).__init__()
        
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
    

# 用rMAP优化的贝叶斯神经网络模型
class rMAP_BayesModel(BaseNet):
    def __init__(self, layers, lamda=1e-2, lr=0.01, data_noise_scale=9, param_noise_scale=0.1,  device='cpu'):               # 网络结构，先验分布，学习率，gpu或cpu
        super(rMAP_BayesModel, self).__init__()
        self.layers = layers
        self.lamda = lamda
        self.lr = lr
        self.data_noise_scale = data_noise_scale
        self.param_noise_scale = param_noise_scale
        self.device = device
        self.model_dir = './model/rmap_net_samples'
        self.model_path = self.model_dir + '/rmap_bayesmodel_0.pth'           # 模型保存路径
        self.net_id = 0
        self.epoch = 0
        self.theta_noise = torch.Tensor()
        
        self.net = rMAP_BayesNet(self.layers)
        self.net.to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), self.lr)
        self.llhd = isotropic_gauss_loglike    # 这里可以换成下方的loss_fn函数，舍弃掉常数项减少计算量

        mkdir(self.model_dir)   # 创建模型保存文件夹

        # 训练日志
        self.loss_log = []
        self.err_log = []
        self.comp_log = []
        self.llhd_log = []

        print("layers: ", self.layers)

    # def llhd(self, y, output):
    #     return (-0.5*(y-output)**2).sum()

    def __sample_theta_noise__(self):
        return self.param_noise_scale*torch.randn(self.get_net_parameters())

    def comp(self, exclude_names=('out_noise//',)):
        l2_reg = 0.0
        param_id = 0
        param_num = 0
        for name, param in self.net.named_parameters():
            if param.requires_grad and not any(ex in name for ex in exclude_names):
                param_num = len(param.view(-1))
                param_noise = self.theta_noise[param_id:param_id+param_num]
                param_noise = param_noise.view(param.shape)
                l2_reg += torch.sum((param-param_noise)** 2)
                param_id += param_num
        return self.lamda*l2_reg
    
    def total_loss(self, y, output):
        output_noise = 1e-6 + F.softplus(self.net.out_noise, beta=1, threshold=20)
        log_prior = sum(self.prior.loglike(p) for p in self.net.parameters() if p.requires_grad)
        log_llhd = self.llhd(y, output, output_noise)
        return -log_prior -log_llhd


    # 用rMAP算法采样模型参数
    def train(self, train_dataset, epochs, n_samples, burn_in_samples):
        Nbatch = len(train_dataset)       # 训练集的batch数，小批量训练时计算复杂性代价comp需要除以batch数

        for i in range(n_samples+burn_in_samples):
            # 数据增加扰动
            train_dataset_noise = train_dataset
            y_noise = self.data_noise_scale*np.random.normal(0, 1, size=(train_dataset.y.shape[0], train_dataset.y.shape[1]))
            train_dataset_noise.y = train_dataset.y + y_noise
            # 模型参数先验增加扰动
            self.theta_noise = self.__sample_theta_noise__()

            for self.epoch in range(epochs):
                train_loader = iter(train_dataset_noise)
                # 记录每个epoch的平均loss
                loss_epoch = 0
                error_epoch = 0
                comp_epoch = 0
                llhd_epoch = 0
                for x,y in train_loader:
                    x, y = torch.from_numpy(x).type(torch.float32), torch.from_numpy(y).type(torch.float32)
                    x, y = x.to(self.device), y.to(self.device)

                    self.optimizer.zero_grad()

                    output = self.net(x)

                    output_noise = 1e-6 + F.softplus(self.net.out_noise, beta=1, threshold=20)
                    loss_comp = self.comp()
                    loss_llhd = self.llhd(y, output, output_noise)

                    loss = loss_comp - loss_llhd
                    loss.backward()
                    self.optimizer.step()

                    # error_epoch += F.mse_loss(output, y).item()
                    error_epoch += (torch.norm(output-y) / torch.norm(y)).item()
                    loss_epoch += loss.item()
                    comp_epoch += loss_comp.item()
                    llhd_epoch += loss_llhd.item()

                loss_epoch = loss_epoch / Nbatch
                error_epoch = error_epoch / Nbatch
                comp_epoch = comp_epoch / Nbatch
                llhd_epoch = llhd_epoch / Nbatch

                if self.epoch % 100 == 0:
                    print('Netid: %d, Epoch: %d, Loss: %.4f, Error: %.4f' % (self.net_id, self.epoch, loss_epoch, error_epoch))

                if i == n_samples-1:
                    self.loss_log.append(loss_epoch)
                    self.err_log.append(error_epoch)
                    self.comp_log.append(comp_epoch)
                    self.llhd_log.append(llhd_epoch)

            if i >= burn_in_samples:
                self.model_path = self.model_dir + '/rmap_bayesmodel_%d' % self.net_id + '.pth'
                self.save(self.model_path)
                self.net_id += 1

    
    # 输入x，输出预测值的均值和方差
    def predict(self, x):
        x = torch.from_numpy(x).type(torch.float32)
        x = x.to(self.device)
        output = self.net(x)
        output_noise = 1e-6 + F.softplus(self.net.out_noise, beta=1, threshold=20)

        return output, output_noise
    