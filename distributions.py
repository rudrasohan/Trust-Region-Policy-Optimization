import math
import torch
import numpy as np
import torch.functional as F
import torch.distributions as P

class Distribution(object):

    def sample(self):
        raise NotImplementedError

    def get_param(self):
        raise NotImplementedError
    
    def logli(self):
        raise NotImplementedError
    
    def kl_div(self, other):
        raise NotImplementedError

    def logli_ratio(self, other, val):
        logli_new = self.logli(val)
        logli_old = other.logli(val)
        return (logli_new - logli_old).exp()


class DiagonalGaussian(Distribution):

    def __init__(self, mean, log_std):
        self.mean = mean
        self.log_std = log_std
        self.normal = P.Normal(self.mean, (self.log_std.exp()))
        self.diagn = P.Independent(self.normal, 1)

    def sample(self):
        return self.diagn.sample()

    def get_param(self):
        return dict(mean=self.mean, log_std=self.log_std)

    def logli(self, val):
        #zs = (val - self.mean) * (-self.log_std).exp()
        #std1 = self.log_std.numpy()
        #mean1 = self.mean.numpy()
        #zs = zs.numpy()
        #print(mean1.shape[-1])
        #final = - np.sum(std1, axis=-1) - 0.5 * np.sum(zs**2, axis=-1) - 0.5 * mean1.shape[-1] * np.log(2 * np.pi)
        #print(np.exp(final))
        #var = self.log_std.exp().pow(2)
        #log_density = -(val - self.mean).pow(2) / (2.0 * var) - 0.5 * math.log(2.0 * math.pi) - self.log_std
        #return log_density.sum(1)
        return self.diagn.log_prob(val)

    #@register_kl(P.Independent, P.Independent)
    def kl_div(self, other):
        deviations = (other.log_std - self.log_std)
        d1 = (2.0 * self.log_std).exp()
        d2 = (2.0 * other.log_std).exp()
        sqmeans = (self.mean - other.mean).pow(2)
        d_KL = (sqmeans + d1 - d2) / (2.0 * d2 + 1e-8) + deviations
        d_KL = d_KL.sum(1)
        d_KL = torch.squeeze(d_KL)
        return d_KL

    def entropy(self):
        return self.diagn.entropy()



def distributions_test():
    torch.manual_seed(60)
    mean = torch.Tensor([[0, 0, 0],[1, 2, 3]])
    std = torch.Tensor([[1.0, 1.0, 1.0],[1.0, 0.0, 1.0]])
    #sampl = mean
    #mean = torch.Tensor([1,2,3])
    #std = torch.Tensor([1, 1, 1])
    dist1 = DiagonalGaussian(mean, std)
    '''
    std2 = torch.eye(3)
    dist2 = P.MultivariateNormal(mean, std2)
    assert dist1.sample().shape == dist2.sample().shape
    assert dist1.logli(torch.Tensor([0,0,0])) == dist2.log_prob(torch.Tensor([0,0,0]))
    dist3 = DiagonalGaussian(mean, std)
    assert dist1.kl_div(dist3) == torch.tensor([0.])
    assert dist1.logli_ratio(dist3, torch.Tensor([1,3,3])) == torch.tensor([1.])
    print(dist1.entropy())'''
    #std2 = torch.eye(3)
    print(dist1.sample())
    print(dist1.logli(mean).exp())
    #d_test = P.MultivariateNormal(mean, std2)
    #d_test1 = P.Normal(0.0, 1.0)
    #print(d_test1.log_prob(0.0).exp())
    #print(d_test.log_prob(mean).exp())


#distributions_test()