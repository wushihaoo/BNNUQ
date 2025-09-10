import torch
from torch.optim import Optimizer
from numpy.random import gamma


# 与HMC对应，用动量作为更新参数
class SGHMC_Optimizer_Momentum(Optimizer):
    def __init__(self, params, step_size=0.01, friction=0.05):
        
        defaults = dict(step_size=step_size, friction=friction)
        super(SGHMC_Optimizer_Momentum, self).__init__(params, defaults)

    def step(self, resample_r=False):
        
        for group in self.param_groups:
            step_size = group['step_size']
            friction = group['friction']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)

                r = state['momentum']
                # 重新采样动量作为马尔科夫链的初始状态
                if resample_r:
                    r = torch.randn_like(p.data)

                # 采样噪声项
                noise = torch.randn_like(p.data) * (2.0*step_size*friction)**0.5

                # 更新动量
                r = (1-step_size*friction)*r - step_size*p.grad + noise

                # 更新参数
                # p.data.add_(r, alpha=step_size)
                p.data = p.data + step_size*r


# 与SGD对应，用速度作为更新参数，质量矩阵为I，对B的估计为0
class SGHMC_Optimizer_Simple(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.01):  # 学习率一般设置为[0.01,0.1]/数据集大小，alpha动量项系数一般设置为[0.01,0.1]，beta此处设置为0，M质量矩阵设置为I
        
        defaults = dict(lr=lr, alpha=alpha)
        super(SGHMC_Optimizer_Simple, self).__init__(params, defaults)

    def step(self, resample_v=False):
        
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['v'] = torch.zeros_like(p.data)

                v = state['v']
                # 重新采样动量作为马尔科夫链的初始状态
                if resample_v:
                    v = torch.randn_like(p.data) * (lr)**0.5

                # 噪声项
                noise = torch.randn_like(p.data) * (2.0*lr*alpha)**0.5

                # 更新速度
                v = (1-alpha)*v - lr*p.grad + noise

                # 更新参数
                p.data.add_(v)


# zxs
class SGHMC_Optimizer(Optimizer):
    def __init__(self, params, lr=0.01, base_C=0.05, prior_sigma=0.1, alpha0=10, beta0=10):  # 学习率一般设置为[0.01,0.1]/数据集大小
        
        self.eps = 1e-6
        self.alpha0 = alpha0
        self.beta0 = beta0

        if prior_sigma == 0:
            self.weight_decay = 0
        else:
            self.weight_decay = 1.0/(prior_sigma**2)

        defaults = dict(lr=lr, base_C=base_C)
        super(SGHMC_Optimizer, self).__init__(params, defaults)

    def step(self, burn_in=True, resample_v=False, resample_prior=False):
        
        for group in self.param_groups:
            lr = group['lr']
            base_C = group['base_C']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p] 
                if len(state) == 0:
                    state['tau'] = torch.ones_like(p.data)
                    state['g'] = torch.ones_like(p.data)
                    state['V_hat'] = torch.ones_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['weight_decay'] = self.weight_decay

                if resample_prior:
                    alpha = self.alpha0 + 0.5*p.data.numel()
                    beta = self.beta0 + 0.5*((p.data**2).sum().item())
                    gamma_sample = gamma(alpha, scale=1.0/beta)
                    state['weight_decay'] = gamma_sample

                tau, g, V_hat, v, weight_decay = state['tau'], state['g'], state['V_hat'], state['v'], state['weight_decay']             

                dp = p.grad
                if weight_decay != 0:
                    dp.add_(p.data, alpha=weight_decay)

                if burn_in:
                    tau.add_(-tau*(g**2)/(V_hat+self.eps) + 1)
                    tau_inv = 1. / (tau+self.eps)
                    g.add_(-tau_inv*g + tau_inv*dp)
                    V_hat.add_(-tau_inv*V_hat + tau_inv*(dp**2))

                V_hat_sqrt = torch.sqrt(V_hat+self.eps)
                V_hat_sqrt_inv = 1. / (V_hat_sqrt+self.eps)

                # 重新采样动量作为马尔科夫链的初始状态
                if resample_v:
                    v = torch.randn_like(p.data) * lr*(V_hat_sqrt_inv)**0.5

                # 噪声项
                noise = torch.randn_like(p.data) * (torch.clamp(2.0*(lr**2)*V_hat_sqrt_inv*base_C-lr**4, min=lr**8))**0.5

                # 更新速度
                v.add_(-(lr**2)*V_hat_sqrt_inv*dp - base_C*v + noise)

                # 更新参数
                p.data.add_(v)