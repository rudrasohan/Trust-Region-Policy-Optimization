from tqdm import tqdm
import gym
#import gym.monitoring as monitor
import math
import numpy as np
import torch
from torch.autograd import Variable
#from models import GaussianMLPPolicy

def sample_trajectories(env, policy, num_samples):

    trajs = dict(state=[], actions=[], rewards=[], next_state=[], dist=dict(means=[], log_std=[]), done=[])
    collected = 0

    progress = tqdm(total=num_samples)
    s_0 = env.reset()
    s_0 = torch.from_numpy(s_0)#.unsqueeze(0)
    while collected < num_samples:
        
        action, dist = policy.actions(s_0)
        #print(action)
        s_1, r, done, info = env.step(action)
        trajs["state"].append(s_0.numpy())
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
            s_0 = torch.from_numpy(s_0)#.unsqueeze(0)
        else:
            s_0 = torch.from_numpy(s_1)#.unsqueeze(0)

    progress.close()
    return trajs


def compute_advantage_returns(trajs, baseline, discount, gae_lambda):
    
    obs = np.asarray(trajs['state'])
    rewards = np.asarray(trajs['rewards'])
    dones = np.asarray(trajs['done'])
    values = baseline.predict(torch.from_numpy(obs))

    prev_return = 0.0#values.data[-1]
    prev_value = 0.0
    prev_adv = 0.0

    returns = torch.zeros(rewards.shape)
    deltas = torch.zeros(rewards.shape)
    advantages = torch.zeros(rewards.shape)

    for i in reversed(range(rewards.shape[0])):
        returns[i] = rewards[i] + discount * dones[i] * prev_return
        deltas[i] = rewards[i] + discount * dones[i] * prev_value - values.data[i]
        advantages[i] = deltas[i] + (gae_lambda * discount) * prev_adv * dones[i]

        prev_return = returns[i]
        prev_value = values[i].data
        prev_adv = advantages[i]

    trajs['dist']['means'] = torch.stack(trajs['dist']['means']).data.clone()
    trajs['dist']['log_std'] = torch.stack(trajs['dist']['log_std']).data.clone()
    trajs['actions'] = torch.stack(trajs['actions'])#.data.clone()
    trajs['returns'] = returns#.data.clone()
    trajs['advantages'] = (advantages - advantages.mean())/ (advantages.std() + 1e-8) 
    trajs['baselines'] = values.data.clone()


def get_flat_params(params):
    flat_params = []
    for param in params:
        flat_params.append(param.view(-1))
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
        #print(param.grad)
        grads.append(param.grad.view(-1))
    flat_grad = torch.cat(grads)
    return flat_grad
    

def normal_log_density(x, mean, log_std):
    std = log_std.exp()
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1)


