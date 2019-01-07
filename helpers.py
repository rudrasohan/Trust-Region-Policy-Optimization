from tqdm import tqdm
import gym
#import gym.monitoring as monitor
import math
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
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

    trajs = dict(state=[], actions=[], rewards=[], next_state=[], act_prob=[], done=[])
    collected = 0
    running_state = ZFilter((env.observation_space.shape[0],), clip=5)
    progress = tqdm(total=num_samples)
    s_0 = env.reset()
    #s_0 = torch.from_numpy(s_0)#.unsqueeze(0)
    s_0 = running_state(s_0)
    #s_0 = torch.from_numpy(s_0)
    step_count = 0
    rew_count = 0.0
    rew_batch = 0.0
    num_ep = 0
    while collected < num_samples:
        
        action, logp = policy.actions(s_0)
        s_1, r, done, info = env.step(action)
        s_1 = running_state(s_1)
        rew_count += r
        trajs["state"].append(s_0)
        trajs["actions"].append(action)
        trajs["rewards"].append(r)
        trajs["next_state"].append(s_1)
        trajs["act_prob"].append(logp)
        #trajs["dist"]["means"].append(dist["mean"])
        #trajs["dist"]["log_std"].append(dist["log_std"])
        collected += 1
        step_count += 1
    
        #if (collected+1)%1000 == 0:
            #print("GO")
            #done = True
        trajs["done"].append(int(not done))
        
        progress.update(1)
        if done:
            #print("STEP COUNT",step_count," TOTAL COUNT", collected)
            #print("REW COUNT",rew_count)
            rew_batch += rew_count
            num_ep += 1
            step_count = 0
            rew_count = 0.0
            s_0 = env.reset()
        else:
            s_0 = s_1

    print("TOTAL REW", rew_batch)
    print("AVERAGE REWARD", rew_batch/num_ep,",",num_ep)

    progress.close()
    return trajs


def compute_advantage_returns(trajs, baseline, discount, gae_lambda):
    
    obs = np.asarray(trajs['state'])
    rewards = torch.Tensor(np.asarray(trajs['rewards']))
    dones = np.asarray(trajs['done'])
    values = baseline.predict(obs).data
    prev_return = 0.0
    prev_value = 0.0
    prev_adv = 0.0

    returns = torch.zeros(rewards.size(0),1)
    deltas = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    for i in reversed(range(rewards.shape[0])):
        returns[i][0] = rewards[i] + discount * dones[i] * prev_return
        deltas[i] = rewards[i] + discount * dones[i] * prev_value - values[i][0].item()
        advantages[i] = deltas[i] + (gae_lambda * discount) * prev_adv * dones[i]

        prev_return = returns[i][0]
        prev_value = values[i][0].item()
        prev_adv = advantages[i]

    trajs['actions'] = torch.stack(trajs['actions'])
    trajs['act_prob'] = torch.stack(trajs['act_prob'])
    trajs['returns'] = returns
    trajs['advantages'] = (advantages - advantages.mean())/ (advantages.std() + 1e-8) 
    trajs['baselines'] = values

def get_flat_params(model):
    flat_params = parameters_to_vector(model.parameters())
    return flat_params

def set_flat_params(model, flat_params):
    vector_to_parameters(flat_params, model.parameters())
    
def get_flat_grads(model):
    flat_grad = parameters_to_vector([v.grad for v in model.parameters()])
    return flat_grad
    

def normal_log_density(x, mean, log_std):
    std = log_std.exp()
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1)


