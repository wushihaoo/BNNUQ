from src.priors import *
from src.base_net import *

import torch
import torch.nn as nn
import torch.nn.functional as F


# %% 神经网络模型
# 贝叶斯线性层
class VI_BayesLayer(nn.Module):
    def __init__(self, n_in, n_out, prior_class):                 # 输入维度，输出维度，先验分布的类
        super(VI_BayesLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior = prior_class

        self.w_mu = nn.Parameter(torch.Tensor(self.n_out, self.n_in).uniform_(-1.0, 1.0))   # 模型参数w服从的变分分布（近似后验分布）的均值
        self.w_rho = nn.Parameter(torch.Tensor(self.n_out, self.n_in).uniform_(-5.0, 5.0))     # 模型参数w服从的变分分布的标准差的softplus参数

        self.b_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-1.0, 1.0))              # 模型参数b服从的变分分布的均值
        self.b_rho = nn.Parameter(torch.Tensor(self.n_out).uniform_(-5.0, 5.0))                # 模型参数b服从的变分分布的标准差的softplus参数

    def forward(self, x, sample=True):
        # 模型训练时，需要采样估计期望
        if sample:
            # 重参数化采样（通过采样eps然后经过变换得到变分分布的采样，如果直接用变分分布参数进行采样得到值进行前向传播，在后向传播时无法对变分分布参数进行梯度计算）
            eps_w = self.w_mu.data.new(self.w_mu.size()).normal_()
            eps_b = self.b_mu.data.new(self.b_mu.size()).normal_()

            # 变分分布的标准差，避免出现负值
            std_w = 1e-6 + F.softplus(self.w_rho, beta=1, threshold=20)
            std_b = 1e-6 + F.softplus(self.b_rho, beta=1, threshold=20)

            # 变分分布采样的模型参数
            w = self.w_mu + eps_w * std_w
            b = self.b_mu + eps_b * std_b

            # 前向传播
            output = F.linear(x, w, b)

            # 蒙特卡洛法计算变分分布和先验分布的对数似然，用于估计KL散度（lqw-lpw）
            lqw = isotropic_gauss_loglike(w, self.w_mu, std_w) + isotropic_gauss_loglike(b, self.b_mu, std_b)
            lpw = self.prior.loglike(w) + self.prior.loglike(b)
            return output, lqw, lpw
        # 模型验证时，直接输出最大后验的点估计
        else:
            output = F.linear(x, self.w_mu, self.b_mu)
            return output, 0, 0
        

# 局部重参数化，只有所有模型参数都是独立高斯分布时才成立
class VI_BayesLayer_LocalRepra(nn.Module):
    def __init__(self, n_in, n_out, prior_class):                 # 输入维度，输出维度，先验分布的类
        super(VI_BayesLayer_LocalRepra, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior = prior_class
        self.prior_var = torch.Tensor([self.prior.sigma**2])       # 模型参数w和b先验分布的方差，用于在局部重参数化时计算输出结果z的先验标准差

        self.w_mu = nn.Parameter(torch.Tensor(self.n_out, self.n_in).uniform_(-1.0, 1.0))   # 模型参数w服从的变分分布（近似后验分布）的均值
        self.w_rho = nn.Parameter(torch.Tensor(self.n_out, self.n_in).uniform_(-1.0, 1.0))     # 模型参数w服从的变分分布的标准差的softplus参数

        self.b_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-1.0, 1.0))             # 模型参数b服从的变分分布的均值
        self.b_rho = nn.Parameter(torch.Tensor(self.n_out).uniform_(-1.0, 1.0))                # 模型参数b服从的变分分布的标准差的softplus参数

    def forward(self, x, sample=True):
        # 模型训练时，需要采样估计期望
        if sample:
            # 局部重参数化采样（计算该层输出z的均值和方差，直接对z进行采样，从而减少计算量）
            z_shape = (x.size()[0], self.n_out)         # 该层输出z的形状
            z_eps = torch.Tensor(*z_shape).normal_()

            # 变分分布的标准差，避免出现负值
            std_w = 1e-6 + F.softplus(self.w_rho, beta=1, threshold=20)
            std_b = 1e-6 + F.softplus(self.b_rho, beta=1, threshold=20)

            # 前向传播
            z_mean = F.linear(x, self.w_mu, self.b_mu)         # z的均值
            z_std = torch.sqrt(torch.mm(torch.pow(x,2), torch.pow(std_w, 2).T) + torch.pow(std_b.expand(*z_shape), 2))     # z的标准差
            z = z_mean + z_std * z_eps      # z的采样
            prior_z_std = torch.sqrt(torch.mm(torch.pow(x,2), self.prior_var.expand(self.n_in, self.n_out)) +      # 用w和b的先验分布计算z的先验分布的标准差
                                     self.prior_var.expand(*z_shape)).detach()

            # 计算变分分布和先验分布的对数似然，用于估计KL散度（lqw-lpw）（这里用了z_mean作为先验的均值，目的为保留先验对方差，即不确定性的估计而忽略其对均值的预估，
            # 在非信息先验时建议使用；否则按计算公式，w和b的先验均值都为0时z的先验均值应为0）
            lqw = isotropic_gauss_loglike(z, z_mean, z_std)
            lpw = isotropic_gauss_loglike(z, z_mean, prior_z_std)

            # 另一种局部重参数化时估算KL散度的方法(暂时不对)
            # kld = 0.5 * (2 * torch.log(prior_z_std / z_std) - 1 + (z_std / prior_z_std).pow(2) + ((0 - z_mean) / prior_z_std).pow(2)).sum()
            # lqw = kld
            # lpw = 0
            # https://arxiv.org/abs/1312.6114 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

            return z, lqw, lpw
        # 模型验证时，直接输出最大后验的点估计
        else:
            z_mean = F.linear(x, self.w_mu, self.b_mu)
            return z_mean, 0, 0
            


# 用VI优化的贝叶斯神经网络
class VI_PBayesNet(nn.Module):
    def __init__(self, dnnlayers, bnnlayers, prior_class, init_out_noise=0.0):
        super(VI_PBayesNet, self).__init__()

        self.bnnlayers = nn.ModuleList([VI_BayesLayer_LocalRepra(bnnlayers[i-1], bnnlayers[i], prior_class) for i in range(1, len(bnnlayers))])        
        self.dnnlayers = nn.ModuleList([nn.Linear(dnnlayers[i-1], dnnlayers[i]) for i in range(1, len(dnnlayers))])

        self.dnnact = nn.Tanh()
        self.bnnact = nn.Tanh()

        self.dy_alpha = nn.Parameter(torch.Tensor([1.0]))  # dy缩放系数
        self.out_noise = nn.Parameter(torch.Tensor([init_out_noise]))  # 模型输出的标准差的softplus参数。为计算似然函数，模型输出需是一个分布，
                                                                        # forward输出的值相当于只是这个分布的均值，而其标准差由log_noise决定

    def forward(self, x, sample=True):

        T = x
        for layer in self.dnnlayers:
            T = layer(T)
            if layer != self.dnnlayers[-1]:
                T = self.dnnact(T)

        # B = torch.hstack([x, y])
        B = x
        tlqw = 0                                 # 所有层参数变分分布的对数似然求和
        tlpw = 0                                 # 所有层参数先验分布的对数似然求和
        for layer in self.bnnlayers:
            B, lqw, lpw = layer(B, sample)
            tlqw += lqw
            tlpw += lpw
            if layer != self.bnnlayers[-1]:
                B = self.bnnact(B)
            # B = self.bnnact(B)
            
        output = torch.sum(T * B, dim=-1).unsqueeze(-1)
        # print("output shape: ", output.shape)

        return output, tlqw, tlpw
    

# 用VI优化的贝叶斯神经网络模型
class VI_PBayesModel(BaseNet):
    def __init__(self, dnnlayers, bnnlayers, prior_class=isotropic_gauss_prior(mu=0, sigma=0.1), learning_rate=1e-3, device='cpu'):               # 网络结构，先验分布，学习率，gpu或cpu
        super(VI_PBayesModel, self).__init__()
        self.dnnlayers = dnnlayers
        self.bnnlayers = bnnlayers
        self.prior = prior_class
        self.lr = learning_rate
        self.device = device
        self.model_dir = './model'
        self.model_path = self.model_dir + '/vi_pbayesmodel.pth'           # 模型保存路径

        self.net = VI_PBayesNet(self.dnnlayers, self.bnnlayers, self.prior)
        self.net.to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        self.loss_fn = isotropic_gauss_loglike    # 这里可以换成下方的loss_fn函数，舍弃掉常数项，默认输出的方差为常数（0.5），减少计算量

        mkdir(self.model_dir)   # 创建模型保存文件夹

        # 训练日志
        self.comp_log = []
        self.llhd_log = []
        self.loss_log = []
        self.err_log = []

        self.epoch = 0

        print("dnn layers: ", self.dnnlayers)
        print("bnn layers: ", self.bnnlayers)

    # def loss_fn(self, y, output, output_noise):
    #     return 1e-3*(-0.5*(y-output)**2).sum()


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
                    output, tlqw, tlpw = self.net(x, sample=True)
                    output_noise = 1e-6 + F.softplus(self.net.out_noise, beta=1, threshold=20)
                    comp_i = (tlqw - tlpw) / Nbatch
                    llhd_i = self.loss_fn(y, output, output_noise)
                    comp += comp_i
                    llhd += llhd_i
                comp /= mc_samples
                llhd /= mc_samples

                loss = comp - llhd
                # loss = -llhd
                loss.backward()
                self.optimizer.step()
                self.schedule.step()
                self.lr = self.optimizer.param_groups[0]['lr']

                error_epoch += F.mse_loss(output, y).item()
                loss_epoch += loss.item()
                comp_epoch += comp.item()
                llhd_epoch += llhd.item()

            loss_epoch = loss_epoch / Nbatch
            comp_epoch = comp_epoch / Nbatch
            llhd_epoch = llhd_epoch / Nbatch
            error_epoch = error_epoch / Nbatch

            if self.epoch % 500 == 0:
                print('Epoch: %d, Loss: %.4f, Comp: %.4f, LLHD: %.4f, Error: %.4f' % (self.epoch, loss_epoch, comp_epoch, llhd_epoch, error_epoch))
                print('self.dy_alpha: %.4f, self.out_noise: %.4f' % (self.net.dy_alpha.item(), self.net.out_noise.item()))

                if error_epoch < eps:
                    eps = error_epoch
                    self.save(self.model_path)
                    print('Best model saved! Best Error: %.4f\n' % eps)

            self.comp_log.append(comp_epoch)
            self.llhd_log.append(llhd_epoch)
            self.loss_log.append(loss_epoch)
            self.err_log.append(error_epoch)


    # 点估计，输出最大后验估计的预测值，以及预测误差
    def test_point_estimate(self, test_dataset):
        x, y = test_dataset.x, test_dataset.y
        x, y = torch.from_numpy(x).type(torch.float32), torch.from_numpy(y).type(torch.float32)
        x, y = x.to(self.device), y.to(self.device)
        output, _, _ = self.net(x, sample=False)
        error = F.mse_loss(output, y).item()
        print('Error: %.4f' % error)

        return output, error
    

    # 分布估计，输出预测值的均值和方差，以及预测误差
    def test_distribution_estimate(self, test_dataset, mc_samples=100):
        x, y = test_dataset.x, test_dataset.y
        x, y = torch.from_numpy(x).type(torch.float32), torch.from_numpy(y).type(torch.float32)
        x, y = x.to(self.device), y.to(self.device)
        output = []
        for i in range(mc_samples):
            output_i, _, _ = self.net(x, sample=True)
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
            output_i, _, _ = self.net(x, sample=True)
            output.append(output_i)
        output = torch.stack(output, dim=0)
        output_mean = torch.mean(output, dim=0)
        output_var = torch.var(output, dim=0)
        output_noise = 1e-6 + F.softplus(self.net.out_noise, beta=1, threshold=20)

        return output_mean, output_var, output_noise
    