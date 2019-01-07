import torch 
import gym
import numpy as np
from torch.autograd import grad
from models import GaussianMLPPolicy, MLPBaseline
from helpers import sample_trajectories
#from trpo import trpo_step,trpo_test
from gym import wrappers
from trpo_agent import TRPO
import logging
from helpers import compute_advantage_returns, sample_trajectories



def run_experiments():
    env = gym.make("BipedalWalker-v2")
    #env = wrappers.Monitor(env, "/home/sohan/gym-results", force=True)
    print(env.action_space)
    policy = GaussianMLPPolicy(env.observation_space, env.action_space)
    baseline = MLPBaseline(env.observation_space, env.action_space)
    agent = TRPO(env, policy, baseline)
    #policy.network#.cuda()
    #baseline.value#.cuda()
    #log = logger
    print(policy.network)
    print(baseline.value)
    #print("COUNT:", policy.count)
    #trpo_test()
    for i in range(100):
        agent.step()
        


torch.set_default_tensor_type('torch.DoubleTensor')
run_experiments()