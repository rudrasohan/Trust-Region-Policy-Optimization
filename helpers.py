from tqdm import tqdm
import gym
#import gym.monitoring as monitor
import math
import numpy as np
import torch
from torch.autograd import Variable
#from models import GaussianMLPPolicy

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

class ZFilter:
    """
    y = (x-mean)/std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape

def sample_trajectories(env, policy, num_samples):

    trajs = dict(state=[], actions=[], rewards=[], next_state=[], dist=dict(means=[], log_std=[]), done=[])
    collected = 0
    running_state = ZFilter((env.observation_space.shape[0],), clip=5)
    progress = tqdm(total=num_samples)
    s_0 = env.reset()
    #s_0 = torch.from_numpy(s_0)#.unsqueeze(0)
    s_0 = running_state(s_0)
    #s_0 = torch.from_numpy(s_0)
    while collected < num_samples:
        
        action, dist = policy.actions(s_0)
        s_1, r, done, info = env.step(action)
        #env.render()
        s_1 = running_state(s_1)
        trajs["state"].append(s_0)
        trajs["actions"].append(action)
        trajs["rewards"].append(r)
        trajs["next_state"].append(s_1)
        trajs["dist"]["means"].append(dist["mean"])
        trajs["dist"]["log_std"].append(dist["log_std"])
        trajs["done"].append(int(not done))
        collected += 1
        progress.update(1)
        if done:
            s_0 = env.reset()
        else:
            s_0 = s_1

    progress.close()
    return trajs


def compute_advantage_returns(trajs, baseline, discount, gae_lambda):
    
    obs = np.asarray(trajs['state'])
    rewards = torch.tensor(np.asarray(trajs['rewards']))
    dones = np.asarray(trajs['done'])
    #print("DONES",dones.shape)
    values = baseline.predict(obs)

    prev_return = 0.0#values.data[-1]
    prev_value = 0.0
    prev_adv = 0.0

    returns = torch.zeros(rewards.size(0),1)
    deltas = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    for i in reversed(range(rewards.shape[0])):
        #print(i)
        returns[i] = rewards[i] + discount * dones[i] * prev_return
        #print("HAHHAHHAHHAHAHAHAHHAHHAHAHHAHHA", discount * dones[i] * prev_value - values.data[i][0])
        deltas[i] = rewards[i] + discount * dones[i] * prev_value - values.data[i][0]
        advantages[i] = deltas[i] + (gae_lambda * discount) * prev_adv * dones[i]

        prev_return = returns[i][0]
        prev_value = values[i][0].data
        prev_adv = advantages[i]
        #print("PREV_R", prev_return)
        #print("PREV_V", prev_value)
        #print("PREV_A", prev_adv)
        #print("RETURNS:{}".format(returns[i]))

    trajs['dist']['means'] = torch.stack(trajs['dist']['means']).data.detach()
    trajs['dist']['log_std'] = torch.stack(trajs['dist']['log_std']).data.detach()
    trajs['actions'] = torch.stack(trajs['actions'])
    trajs['returns'] = returns
    trajs['advantages'] = (advantages - advantages.mean())/ (advantages.std() + 1e-8) 
    trajs['baselines'] = values.data.detach()


def get_flat_params(params):
    flat_params = []
    for param in params:
        flat_params.append(param.data.view(-1))
    flat_params = torch.cat(flat_params)
    return flat_params

def set_flat_params(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:(prev_ind + flat_size)].view(param.size()))
        prev_ind += flat_size
    
def get_flat_grads(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    flat_grad = torch.cat(grads)
    return flat_grad
    

def normal_log_density(x, mean, log_std):
    std = log_std.exp()
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1)


