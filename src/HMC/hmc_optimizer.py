import torch
from torch.optim import Optimizer

torch.random.manual_seed(1234)

def make_closure(model, input, target, loss_fn):
        def closure():
            # 清空梯度
            model.zero_grad()

            # 前向计算
            output = model(input)
            loss = loss_fn(target, output)

            # 反向传播
            loss.backward()

            return loss

        return closure

# 计算梯度的loss为估计的后验负对数
class HMC_Optimizer_nlp(Optimizer):
    def __init__(self, params, step_size=0.01, n_steps=10):
        
        defaults = dict(step_size=step_size, n_steps=n_steps)
        super(HMC_Optimizer_nlp, self).__init__(params, defaults)

    def step(self, closure=None):
        
        for group in self.param_groups:
            step_size = group['step_size']
            n_steps = group['n_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["accepted"] = 0
                    state["total"] = 0

                # 初始位置
                theta_0 = p.data.clone()
                grad_0 = p.grad.data.clone()
                theta = theta_0.clone()

                # 采样初始动量
                r_0 = torch.randn_like(p.data)
                r = r_0.clone()

                # 初始哈密顿量
                loss_0 = closure()
                U_0 = loss_0.item()
                K_0 = 0.5 * torch.sum(r_0 ** 2).item()

                # Leapfrog积分器
                r -= 0.5*step_size*grad_0
                for i in range(n_steps):
                    theta += step_size*r
                    
                    # 更新梯度
                    p.data.copy_(theta)
                    loss = closure()
                    grad = p.grad.data

                    if i != n_steps - 1:
                        r -= step_size*grad
                r -= 0.5*step_size*grad

                # 反转动量保证HMC可逆性
                # r = -r

                # Metropolis-Hastings接受-拒绝步骤
                U_t = loss.item()
                K_t = 0.5 * torch.sum(r ** 2).item()
                acceptance_ratio = torch.exp(torch.tensor(U_0 + K_0 - U_t - K_t))
                accept = torch.rand(1) < acceptance_ratio
                if accept:
                    p.data.copy_(theta)
                    state["accepted"] += 1
                else:
                    p.data.copy_(theta_0)
                    loss = loss_0
                state["total"] += 1
        return loss
    
    def get_acceptance_rates(self):
        rates = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if state["total"] > 0:
                    rates[id(p)] = state["accepted"] / state["total"]
                else:
                    rates[id(p)] = None
        print("推荐接受率范围： [0.65-0.8]")
        return rates


# 计算梯度的loss为估计的似然负对数，通过引入正则化项来估计后验负对数
class HMC_Optimizer(Optimizer):
    def __init__(self, params, step_size=0.01, n_steps=10, prior_sigma=2.0):

        if prior_sigma == 0:
            self.weight_decay = 0
        else:
            self.weight_decay = 1.0/(prior_sigma**2)
        
        defaults = dict(step_size=step_size, n_steps=n_steps)
        super(HMC_Optimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        
        for group in self.param_groups:
            step_size = group['step_size']
            n_steps = group['n_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["accepted"] = 0
                    state["total"] = 0
                    state["weight_decay"] = self.weight_decay

                weight_decay = state["weight_decay"]
                dp = p.grad
                if weight_decay != 0:
                    dp.add_(p.data, alpha=weight_decay)

                # 初始位置
                theta_0 = p.data.clone()
                grad_0 = dp.data.clone()
                theta = theta_0.clone()

                # 采样初始动量
                r_0 = torch.randn_like(p.data)
                r = r_0.clone()

                # 初始哈密顿量
                loss_0 = closure()
                U_0 = loss_0.item()
                K_0 = 0.5 * torch.sum(r_0 ** 2).item()

                # Leapfrog积分器
                r -= 0.5*step_size*grad_0
                for i in range(n_steps):
                    theta += step_size*r
                    
                    # 更新梯度
                    p.data.copy_(theta)
                    loss = closure()
                    dp = p.grad
                    if weight_decay != 0:
                        dp.add_(p.data, alpha=weight_decay)

                    if i != n_steps - 1:
                        r -= step_size*dp
                r -= 0.5*step_size*dp

                # 反转动量保证HMC可逆性
                # r = -r

                # Metropolis-Hastings接受-拒绝步骤
                U_t = loss.item()
                K_t = 0.5 * torch.sum(r ** 2).item()
                acceptance_ratio = torch.exp(torch.tensor(U_0 + K_0 - U_t - K_t))
                accept = torch.rand(1) < acceptance_ratio
                if accept:
                    p.data.copy_(theta)
                    state["accepted"] += 1
                else:
                    p.data.copy_(theta_0)
                    loss = loss_0
                state["total"] += 1
        return loss
    
    def get_acceptance_rates(self):
        rates = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if state["total"] > 0:
                    rates[id(p)] = state["accepted"] / state["total"]
                else:
                    rates[id(p)] = None
        print("推荐接受率范围： [0.65-0.8]")
        return rates