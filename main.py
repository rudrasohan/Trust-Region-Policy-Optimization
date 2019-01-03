import torch 
import gym
import numpy as np
from torch.autograd import grad
from models import GaussianMLPPolicy, MLPBaseline
from helpers import sample_trajectories
from trpo import trpo_step
import logging


def run_experiments():
    #env = gym.make("MountainCarContinuous-v0")
    env = gym.make("BipedalWalker-v2")
    print(env.action_space)
    policy = GaussianMLPPolicy(env.observation_space, env.action_space)
    baseline = MLPBaseline(env.observation_space, env.action_space)
    #log = logger
    print(policy.network)
    #print("COUNT:", policy.count)
    for i in range(100):
        ep_len = 5000
        paths = sample_trajectories(env, policy, ep_len)
        #print("Reward = {}".format(np.(paths["rewards"])))'
        sumi = 0
        avg = []
        for num in range(len(paths["rewards"])):
            if paths["done"][num]:
                sumi += paths["rewards"][num]
            else:
                avg.append(sumi)
                sumi = 0
        print("Reward = {}".format(np.mean(avg)))
        #baseline.update(paths)
        trpo_step(policy, baseline, paths)
        #baseline.update(paths)
        #print("COUNT:", policy.count)
        


torch.set_default_tensor_type('torch.DoubleTensor')
run_experiments()