"""Model Definations for trpo."""

import gym
import numpy as np
import torch
import time
import scipy.optimize
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from distributions import DiagonalGaussian
from helpers import get_flat_params, set_flat_params, get_flat_grads
#from helpers import sample_trajectories, compute_advantage_returns, get_flat_params

class Model(object):
    """Generic Model Template"""
    
    def __init__(self,
                 observation_space,
                 action_space,
                 **kwargs):
        #super(Model).__init__(**kwargs)
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_dim = None
        self.act_dim = None

        if isinstance(self.observation_space, gym.spaces.Box):
            self.obs_dim = np.prod(self.observation_space.shape)
        else:
            self.obs_dim = self.observation_space.n
        
        if isinstance(self.action_space, gym.spaces.Box):
            self.act_dim = np.prod(self.action_space.shape)
        else:
            self.act_dim = self.action_space.n


class MLP_Policy(nn.Module):
    """MLP model fo the network"""

    def __init__(self, input_dim, output_dim, name, **kwargs):
        super(MLP_Policy, self).__init__()
        self.name = name
        self.use_new_head = False
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)
        if bool(kwargs):
            self.use_new_head = kwargs["use_new_head"]
            self.fc4 = nn.Linear(64, output_dim)

        else:
            self.log_std = nn.Parameter(torch.zeros(output_dim))
            #print(self.log_std.size())
        #self.bn1 = nn.BatchNorm1d(64)
        #self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        #print(self.fc1(x))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.fc3(x)
        if self.use_new_head:
            std = self.fc4(x)
        else:
            std = self.log_std.expand(mean.size())
        #print(mean)
        return mean, std


class MLP_Value(nn.Module):
    """MLP model fo the network"""

    def __init__(self, input_dim, output_dim, name, **kwargs):
        super(MLP_Value, self).__init__()
        self.name = name
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        #print(self.fc1(x))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        out = self.fc3(x)
        return out


class GaussianMLPPolicy(Model):
    """Gaussian MLP Policy"""
    def __init__(self, observation_space, action_space, **kwargs):
        Model.__init__(self, observation_space, action_space, **kwargs)
        #self.mean_network = MLP(self.obs_dim, self.act_dim, "mean").type(torch.float64)
        self.std_net = None
        #self.std_network = None
        #print(kwargs)
        if bool(kwargs):
            self.std_net = kwargs["use_std_net"]
        if self.std_net:
            self.network = MLP_Policy(self.obs_dim, self.act_dim, "MLP_policy", use_new_head=True)#.type(torch.float64)
        else:
            self.network = MLP_Policy(self.obs_dim, self.act_dim, "MLP_policy")#.type(torch.float64)

    def actions(self, obs):
        obs = torch.from_numpy(obs)
        mean, log_std = self.network(obs)            
        dist = DiagonalGaussian(mean, log_std)
        return dist.sample(), dist.get_param()

    def clear_grads(self):
        self.network.zero_grad()
        

class MLPBaseline(Model):
    """"MLP Baseline"""
    def __init__(self, observation_space, action_space, **kwargs):
        Model.__init__(self, observation_space, action_space, **kwargs)
        self.value = MLP_Value(self.obs_dim, 1, "MLP_baseline")
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.LBFGS(self.value.parameters())

    def predict(self, obs):
        obs = torch.Tensor(obs)
        with torch.no_grad():
            val = self.value(obs)
        return val

    def compute_baseline(self, obs):
        obs = torch.Tensor(obs)
        return self.value(torch.tensor(obs))

    def clear_grads(self):
        self.value.zero_grad()

    def update(self, trajs):
        obs = np.asarray(trajs["state"])
        obs = torch.from_numpy(obs)
        returns = trajs["returns"]
        baselines = trajs["baselines"]
        returns = returns.view((obs.shape[0],1))
        targets =  returns * 0.9 + 0.1 * baselines
        #returns = 
        #targets = Variable(returns)
        #print(targets)
        '''
        def closure():
            self.clear_grads()
            values = self.value(torch.from_numpy(obs))
            self.optimizer.zero_grad()
            loss = self.criterion(values, targets)
            print("LBFGS_LOSS:{}".format(loss))
            loss.backward()
            return loss
            '''

        
        #self.optimizer.step(closure)

        #curr_params = get_flat_params(self.value.parameters()).data.detach().double().numpy()
        curr_flat_params = get_flat_params(self.value.parameters()).double().numpy()
        
        def val_loss_grad(x):
            
            set_flat_params(self.value, torch.tensor(x))
            self.clear_grads()
            
            #for param in self.value.parameters():
                #if param.grad is not None:
                    #print("HHAHAHAHAHHA")
                    #param.grad.data.fill_(0)
            #values_ = #self.value(torch.from_numpy(obs))
            
            values_ = self.value(Variable(obs))
            #print("VALUES",values_.size())
            #print("TARGETS",targets.size())
            #print((values_-targets).size())
            #time1 = time.time()
            vf_loss = (values_ - targets).pow(2).mean()
            #print("LBFGS_LOSS:{}".format(vf_loss))
            #time2 = time.time()
            #print("TIME:{}".format(time2-time1))
            #for param in self.value.parameters():
            #    vf_loss += param.pow(2).sum() * 1e-2
            vf_loss.backward()
            flat_grad = get_flat_grads(self.value)
            
            return (vf_loss.data.double().numpy(), flat_grad.data.double().numpy())

        new_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(val_loss_grad, curr_flat_params, maxiter=25)
        set_flat_params(self.value, torch.Tensor(new_params))
        
        print(opt_info)

        


    


def test_policy_value():
    env = gym.make("MountainCarContinuous-v0")
    policy = GaussianMLPPolicy(env.observation_space, env.action_space, use_std_net=True)
    paths = sample_trajectories(env, policy, 1000)
    print(len(paths["rewards"]))
    baseline = MLPBaseline(env.observation_space, env.action_space)
    compute_advantage_returns(paths, baseline, 0.9, 0.1)
    print(paths.keys())
    baseline.update(paths)
    print(paths['dist'].keys())
    flat_params_mean = get_flat_params(policy.mean_network.parameters())
    flat_params_std = get_flat_params(policy.std_network.parameters())
    print(flat_params)

#test_policy_value()