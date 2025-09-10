from __future__ import division
import numpy as np
import torch


# 计算各向同性（每个维度独立）高斯分布的对数似然
def isotropic_gauss_loglike(x, mu, sigma, do_sum=True):
    cte_term = torch.tensor(-0.5 * np.log(2 * np.pi), dtype=x.dtype, device=x.device)
    det_sig_term = -torch.log(sigma)
    inner = (x - mu) / sigma
    dist_term = -(0.5) * (inner ** 2)

    if do_sum:
        out = (cte_term + det_sig_term + dist_term).sum()  # sum over all weights
        # out = dist_term.sum()
    else:
        out = (cte_term + det_sig_term + dist_term)
        # out = dist_term
    return out


# 创建各向同性高斯分布和计算对数似然
class isotropic_gauss_prior(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

        self.cte_term = -(0.5) * np.log(2 * np.pi)
        self.det_sig_term = -np.log(self.sigma)

    def loglike(self, x, do_sum=True):

        dist_term = -(0.5) * ((x - self.mu) / self.sigma) ** 2
        if do_sum:
            # return (self.cte_term + self.det_sig_term + dist_term).sum()
            return dist_term.sum()
        else:
            # return (self.cte_term + self.det_sig_term + dist_term)
            return dist_term


# 创建拉普拉斯分布和计算对数似然
class laplace_prior(object):
    def __init__(self, mu, b):
        self.mu = mu
        self.b = b

    def loglike(self, x, do_sum=True):
        if do_sum:
            return (-np.log(2 * self.b) - torch.abs(x - self.mu) / self.b).sum()
        else:
            return (-np.log(2 * self.b) - torch.abs(x - self.mu) / self.b)



class spike_slab_2GMM(object):
    def __init__(self, mu1, mu2, sigma1, sigma2, pi):
        self.N1 = isotropic_gauss_prior(mu1, sigma1)
        self.N2 = isotropic_gauss_prior(mu2, sigma2)

        self.pi1 = pi
        self.pi2 = (1 - pi)

    def loglike(self, x):
        N1_ll = self.N1.loglike(x)
        N2_ll = self.N2.loglike(x)

        # Numerical stability trick -> unnormalising logprobs will underflow otherwise
        max_loglike = torch.max(N1_ll, N2_ll)
        normalised_like = self.pi1 * torch.exp(N1_ll - max_loglike) + self.pi2 * torch.exp(N2_ll - max_loglike)
        loglike = torch.log(normalised_like) + max_loglike

        return loglike